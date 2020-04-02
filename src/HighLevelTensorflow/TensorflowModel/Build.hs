{-# LANGUAGE ExplicitForAll    #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedLists   #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Rank2Types        #-}
{-# LANGUAGE RankNTypes        #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TypeFamilies      #-}
module HighLevelTensorflow.TensorflowModel.Build
    ( ModelBuilderFunction
    , OutputLayerColumns
    , BuildSetup (..)
    , scaleWeights
    , BuildM
    , buildModel
    , buildModelWith
    , inputLayer1D
    , inputLayer
    , fullyConnected
    , fullyConnectedLinear
    , fullyConnectedWith
    , trainingByAdam
    , trainingByAdamWith
    , trainingByRmsProp
    , trainingByRmsPropWith
    , trainingByGradientDescent
    , randomParam
    ) where

import           Control.Lens
import           Control.Monad                            (when)
import           Control.Monad.Trans.Class                (lift)
import           Control.Monad.Trans.Reader
import           Control.Monad.Trans.State
import           Data.Default
import           Data.Int                                 (Int64)
import           Data.Maybe                               (isJust, isNothing)
import           Data.String                              (fromString)
import           Data.Text                                (Text, pack)

import qualified TensorFlow.Build                         as TF (addNewOp, evalBuildT,
                                                                 explicitName, opDef,
                                                                 opDefWithName, opType,
                                                                 runBuildT, summaries)
import qualified TensorFlow.Core                          as TF hiding (value)
-- import qualified TensorFlow.GenOps.Core                         as TF (square)
import qualified TensorFlow.BuildOp                       as TF (OpParams)
import qualified TensorFlow.GenOps.Core                   as TF (abs, add, add',
                                                                 approximateEqual,
                                                                 approximateEqual, assign,
                                                                 cast, getSessionHandle,
                                                                 getSessionTensor,
                                                                 identity', identityN',
                                                                 lessEqual, matMul, mul,
                                                                 readerSerializeState,
                                                                 relu, shape, square, sub,
                                                                 tanh, tanh',
                                                                 truncatedNormal)
import qualified TensorFlow.Gradient                      as TF
import qualified TensorFlow.Minimize                      as TF
import qualified TensorFlow.Ops                           as TF (initializedVariable,
                                                                 initializedVariable',
                                                                 placeholder, placeholder',
                                                                 reduceMean, reduceSum,
                                                                 restore, save, scalar,
                                                                 vector,
                                                                 zeroInitializedVariable,
                                                                 zeroInitializedVariable')
import qualified TensorFlow.Tensor                        as TF (Ref (..),
                                                                 collectAllSummaries,
                                                                 tensorNodeName,
                                                                 tensorRefFromName,
                                                                 tensorValueFromName,
                                                                 toBuild, value)


import           HighLevelTensorflow.Minimizer
import           HighLevelTensorflow.OptimizerVariables
import           HighLevelTensorflow.TensorflowModel.Type


type OutputLayerColumns = Int64
type ModelBuilderFunction = forall m . (TF.MonadBuild m) => OutputLayerColumns -> m TensorflowModel

data BuildInfo = BuildInfo
  { _inputName         :: !(Maybe Text)
  , _outputName        :: !(Maybe Text)
  , _labelName         :: !(Maybe Text)
  , _maybeTrainingNode :: !(Maybe TF.ControlNode)
  , _nnVars            :: ![TF.Tensor TF.Ref Float]
  , _trainVars         :: ![TF.Tensor TF.Ref Float]
  , _optimizerVars     :: ![OptimizerVariables]
  , _nrUnitsLayer      :: ![[Int64]]
  , _lastTensor        :: !(Maybe (Int64, TF.Tensor TF.Value Float))
  , _nrLayers          :: !Int
  }
makeLenses ''BuildInfo

-- | Setup for building the ANN.
data BuildSetup =
  BuildSetup
    { _scaleWeights :: !Float -- ^ Scale weights with a factor. This is the first parameter used in `randomParam`.
    }
makeLenses ''BuildSetup

instance Default BuildSetup where
  def = BuildSetup 1


type BuildM m = StateT BuildInfo (ReaderT BuildSetup m)


batchSize :: Int64
batchSize = -1

inputTensorName :: Text
inputTensorName = "input"

layerIdxStartNr :: Int
layerIdxStartNr = 0

labLayerName :: Text
labLayerName = "labels"


inputLayer1D :: (TF.MonadBuild m) => Int64 -> BuildM m ()
inputLayer1D numInputs = inputLayer [numInputs]

inputLayer :: (TF.MonadBuild m) => [Int64] -> BuildM m ()
inputLayer shape = do
  let numInputs = product shape
  input <- lift $ lift $ TF.placeholder' (TF.opName .~ TF.explicitName inputTensorName) [batchSize, numInputs]
  lastTensor .= Just (numInputs, input)
  nrLayers .= layerIdxStartNr
  inputName .= Just inputTensorName

fullyConnected :: (TF.MonadBuild m) => [Int64] -> (TF.OpParams -> TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float) -> BuildM m ()
fullyConnected shape activationFunction = fullyConnectedWith shape (Just activationFunction)

fullyConnectedLinear :: (TF.MonadBuild m) => [Int64] -> BuildM m ()
fullyConnectedLinear shape = fullyConnectedWith shape Nothing

fullyConnectedWith :: (TF.MonadBuild m) => [Int64] -> Maybe (TF.OpParams -> TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float) -> BuildM m ()
fullyConnectedWith shape mActivationFunction = do
  layers <- gets (^. nrLayers)
  inpLayer <- gets (^. inputName)
  when (layers < layerIdxStartNr || isNothing inpLayer) $ error "You must start your model with an input layer"
  trainNode <- gets (^. maybeTrainingNode)
  when (isJust trainNode) $ error "You must create the NN before specifying the training nodes."
  let layerNr = layers + 1
  lastLayer <- gets (^. lastTensor)
  case lastLayer of
    Nothing -> error "No previous layer found on fullyConnected1D. Start with an input layer, see the `input1D` function"
    Just (previousNumUnits, previousTensor) -> do
      let numUnits = product shape
      setup <- lift ask
      hiddenWeights <- lift $ lift $ TF.initializedVariable' (TF.opName .~ fromString ("weights" ++ show layerNr)) =<< randomParam (setup ^. scaleWeights) previousNumUnits [previousNumUnits, numUnits]
      hiddenBiases <- lift $ lift $ TF.zeroInitializedVariable' (TF.opName .~ fromString ("bias" ++ show layerNr)) [numUnits]
      let outName = "out" <> pack (show layerNr)
      hidden <-
        case mActivationFunction of
          Nothing -> lift $ lift $ TF.render $ TF.add' (TF.opName .~ TF.explicitName outName) (previousTensor `TF.matMul` hiddenWeights) hiddenBiases
          Just activationFunction -> do
            let hiddenZ = (previousTensor `TF.matMul` hiddenWeights) `TF.add` hiddenBiases
            lift $ lift $ TF.render $ activationFunction (TF.opName .~ TF.explicitName outName) hiddenZ
      nnVars %= (++ [hiddenWeights, hiddenBiases])
      lastTensor .= Just (numUnits, hidden)
      outputName .= Just outName
      nrLayers += 1
      nrUnitsLayer %= (++ [[previousNumUnits, numUnits], [numUnits]])


trainingByAdam :: (TF.MonadBuild m) => BuildM m ()
trainingByAdam = trainingByAdamWith TF.AdamConfig {TF.adamLearningRate = 0.01, TF.adamBeta1 = 0.9, TF.adamBeta2 = 0.999, TF.adamEpsilon = 1e-8}

trainingByRmsProp :: (TF.MonadBuild m) => BuildM m ()
trainingByRmsProp = trainingByRmsPropWith RmsPropConfig {rmsPropLearningRate = 0.001, rmsPropRho = 0.9, rmsPropMomentum = 0.0, rmsPropEpsilon = 1e-7}


-- TODO: let the user decide the loss function

-- L1: Least absolut deviation LAD
-- L2: Least square error LSE

trainingByAdamWith :: (TF.MonadBuild m) => TF.AdamConfig -> BuildM m ()
trainingByAdamWith adamConfig = trainingBy parseOptRefs (adamRefs' adamConfig)
  where parseOptRefs [lrRef] = AdamRefs lrRef
        parseOptRefs xs = error $ "Unexpected number of returned optimizer refs: " <> show (length xs)

trainingByRmsPropWith :: (TF.MonadBuild m) => RmsPropConfig -> BuildM m ()
trainingByRmsPropWith rmsPropConfig = trainingBy parseOptRefs (rmsPropRefs' rmsPropConfig)
  where parseOptRefs [lrRef] = RmsPropRefs lrRef
        parseOptRefs xs = error $ "Unexpected number of returned optimizer refs: " <> show (length xs)


trainingByGradientDescent  :: (TF.MonadBuild m) => Float -> BuildM m ()
trainingByGradientDescent lr = trainingBy parseOptRefs (gradientDescentRefs lr)
  where parseOptRefs [lrRef] = GradientDescentRefs lrRef
        parseOptRefs xs = error $ "Unexpected number of returned optimizer refs: " <> show (length xs)

trainingBy ::
     (TF.MonadBuild m)
  => ([TF.Tensor TF.Ref Float] -> OptimizerVariables) -- ^ How to save optimizer refs
  -> MinimizerRefs Float                      -- ^ Optimizer to use
  -> BuildM m ()
trainingBy optRefFun optimizer = do
  mOutput <- gets (^. outputName)
  when (isNothing mOutput) $ error "You must specify at least one layer, e.g. fullyConnected1D."
  lastLayer <- gets (^. lastTensor)
  -- Create training action.
  case lastLayer of
    Nothing -> error "No previous layer found in trainingByAdam. Start with an input layer, followed by at least one further layer."
    Just (_, previousTensor) -> do
      weights <- gets (^. nnVars)
      nrUnits <- gets (^. nrUnitsLayer)
      labels <- lift $ lift $ TF.placeholder' (TF.opName .~ TF.explicitName labLayerName) [batchSize]
      let loss = TF.square (previousTensor `TF.sub` labels)
      (trainStep, trVars, minimizerRefs) <- lift $ lift $ minimizeWithRefs optimizer loss weights (map TF.Shape nrUnits)

      let vals = map TF.value weights

      grad <- lift $ lift $ TF.gradients loss vals >>= optimizer weights (map TF.Shape nrUnits) -- TF.identity' (TF.opName .~ "gradients") (TF.gradients loss vals)
      grad' <- lift $ lift $ TF.gradients loss vals
      lift $ lift $ TF.identityN' (TF.opName .~ "gradients")  grad'
      -- >>= minimizer params shapes

      trainVars .= trVars
      maybeTrainingNode .= Just trainStep
      labelName .= Just labLayerName
      lastTensor .= Nothing
      optimizerVars %= (++ [optRefFun minimizerRefs])


buildModel :: (TF.MonadBuild m) => BuildM m () -> m TensorflowModel
buildModel = buildModelWith def

buildModelWith :: (TF.MonadBuild m) => BuildSetup -> BuildM m () -> m TensorflowModel
buildModelWith setup builder = do
  buildInfo <- runReaderT (execStateT builder emptyBuildInfo) setup
  case buildInfo of
    BuildInfo (Just inp) (Just out) (Just lab) (Just trainN) nnV trV optRefs _ _ _ -> return $ TensorflowModel inp out lab trainN nnV trV optRefs
    BuildInfo Nothing _ _ _ _ _ _ _ _ _ -> error "No input layer specified"
    BuildInfo _ Nothing _ _ _ _ _ _ _ _ -> error "No output layer specified"
    BuildInfo _ _ Nothing _ _ _ _ _ _ _ -> error "No training model specified"
    BuildInfo _ _ _ Nothing _ _ _ _ _ _ -> error "No training node specified (programming error in training action!)"
  where
    emptyBuildInfo = BuildInfo Nothing Nothing Nothing Nothing [] [] [] [] Nothing layerIdxStartNr

-- | Create tensor with random values where the stddev depends on the width. Takes an additional scale as first argument
-- which is multiplied with each randomly selected weight.
randomParam :: (TF.MonadBuild m) => Float -> Int64 -> TF.Shape -> m (TF.Tensor TF.Build Float)
randomParam scale width (TF.Shape shape) = (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (scale / sqrt (fromIntegral width))


