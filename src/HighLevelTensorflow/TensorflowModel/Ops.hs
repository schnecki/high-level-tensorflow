{-# LANGUAGE OverloadedLists #-}
module HighLevelTensorflow.TensorflowModel.Ops
    ( setLearningRates
    , getLearningRates
    , forwardRun
    , backwardRun
    , copyValuesFromTo
    , saveModel
    , saveModelWithLastIO
    , restoreModel
    , restoreModelWithLastIO
    , buildTensorflowModel
    ,
    ) where

import           Control.DeepSeq
import           Control.Monad                            (unless, void, zipWithM)
import           Control.Monad.IO.Class                   (MonadIO, liftIO)
import qualified Data.ByteString                          as BS
import qualified Data.ByteString.Char8                    as B8
import           Data.List                                (foldl', genericLength)
import qualified Data.Map.Strict                          as M
import           Data.Maybe                               (fromMaybe, isJust)
import           Data.Serialize
import           Data.Serialize.Text                      ()
import           Data.Text                                (Text)
import qualified Data.Vector                              as V
import           System.Directory
import           System.IO.Temp
import           System.IO.Unsafe

import qualified TensorFlow.Core                          as TF
import qualified TensorFlow.Nodes                         as TF (nodesUnion)
import qualified TensorFlow.Ops                           as TF hiding
                                                                 (initializedVariable,
                                                                 zeroInitializedVariable)
import qualified TensorFlow.Output                        as TF (ControlNode (..),
                                                                 NodeName (..))
import qualified TensorFlow.Tensor                        as TF (Tensor (..),
                                                                 tensorNodeName,
                                                                 tensorRefFromName)

import           TensorFlow.Session

import           HighLevelTensorflow.OptimizerVariables
import           HighLevelTensorflow.TensorflowModel.Type
import           HighLevelTensorflow.Util


type Outputs = [[Float]]        -- ^ 1st level: rows, 2nd level: input nodes
type Inputs = [[Float]]         -- ^ 1st level: rows, 2nd level: input nodes
type Labels = [[Float]]         -- ^ 1st level: rows, 2nd level: input nodes

-- | Todo: make this value a variable setting
trainMaxVal :: Float
trainMaxVal = 0.95

-- | Model name of saved file
modelName :: String
modelName = "model"

-- | Training node name of saved file
trainName :: String
trainName = "train"


-- | Set all learning rates of all optimizers. Use the same order are in @optimizerVariables@ and @getLearningRates@.
setLearningRates :: (MonadIO m) => [Float] -> TensorflowModel' -> SessionT m ()
setLearningRates learningRates model = zipWithM TF.assign lrRefs (map TF.scalar learningRates) >>= TF.run_
  where
    lrRefs = concatMap getLearningRateRefs (optimizerVariables $ tensorflowModel model)

-- | Get all learning rates of all optimizers.
getLearningRates :: (MonadIO m) => TensorflowModel' -> SessionT m [Float]
getLearningRates model = do
  lrValues <- TF.run lrRefs
  return $ map V.head (lrValues :: [V.Vector Float])
  where
    lrRefs = concatMap getLearningRateRefs (optimizerVariables $ tensorflowModel model)

-- | This helper function encodes the input batch.
encodeInputBatch :: Inputs -> TF.TensorData Float
encodeInputBatch xs = TF.encodeTensorData [genericLength xs, genericLength (head' xs)] (V.fromList $ mconcat xs)
  where head' []    = error "head: empty input data in encodeInputBatch"
        head' (x:_) = x

-- | This helper function encodes the labels.
encodeLabelBatch :: Labels -> TF.TensorData Float
encodeLabelBatch xs = TF.encodeTensorData [genericLength xs, genericLength (head' xs)] (V.fromList $ mconcat xs)
  where head' []    = error "head: empty input data in encodeLabelBatch"
        head' (x:_) = x

-- | Forward run the model.
forwardRun :: (MonadIO m) => TensorflowModel' -> Inputs -> SessionT m Outputs
forwardRun model inp =
  let inRef = getRef (inputLayerName $ tensorflowModel model)
      outRef = getRef (outputLayerName $ tensorflowModel model)
      inpT = encodeInputBatch inp
      nrOuts = length inp
   in do res <- V.toList <$> TF.runWithFeeds [TF.feed inRef inpT] outRef
         return $
           -- trace ("res: " ++ show res)
           -- trace ("output: " ++ show (separate (length res `div` nrOuts) res []))
           separateInputRows (length res `div` nrOuts) res []
  where
    separateInputRows _ [] acc = reverse acc
    separateInputRows len xs acc
      | length xs < len = error $ "error in separate (in Tensorflow.forwardRun), not enough values: " ++ show xs ++ " - len: " ++ show len
      | otherwise = separateInputRows len (drop len xs) (take len xs : acc)

-- | Train tensorflow model with checks.
backwardRun :: (MonadIO m) => TensorflowModel' -> Inputs -> Labels -> SessionT m ()
backwardRun model inp lab
  | null inp || any null inp || null lab = error $ "Empty parameters in backwardRun not allowed! inp: " ++ show inp ++ ", lab: " ++ show lab
  | otherwise =
    let inRef = getRef (inputLayerName $ tensorflowModel model)
        labRef = getRef (labelLayerName $ tensorflowModel model)
        inpT = encodeInputBatch inp
        labT = encodeLabelBatch $ map (map (max (-trainMaxVal) . min trainMaxVal)) lab
    in TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT] (trainingNode $ tensorflowModel model)

-- | Copies values from one model to the other.
copyValuesFromTo :: (MonadIO m) => TensorflowModel' -> TensorflowModel' -> SessionT m ()
copyValuesFromTo from to = do
  let fromVars = neuralNetworkVariables $ tensorflowModel from
      toVars = neuralNetworkVariables $ tensorflowModel to
  if length fromVars /= length toVars
    then error "cannot copy values to models with different length of neural network variables"
    else void $ zipWithM TF.assign toVars fromVars >>= TF.run_

-- | Save the model with the last input/output values. Also see @saveModel@.
saveModelWithLastIO :: (MonadIO m) => TensorflowModel' -> SessionT m TensorflowModel'
saveModelWithLastIO model =
  case lastInputOutputTuple model of
    Nothing     -> error "No last IO in saveModelWithLastIO"
    Just (i, o) -> saveModel model [i] [o]

-- | Save the model to disk. You can set the output directory by setting @checkpointBaseFileName@, in case it is Nothing
-- a temporary directory will be created and set in the @TensorflowModel'@.
saveModel :: (MonadIO m) => TensorflowModel' -> Inputs -> Labels -> SessionT m TensorflowModel'
saveModel model inp lab = do
  let tempDir = getCanonicalTemporaryDirectory >>= flip createTempDirectory ""
  basePath <- maybe (liftIO tempDir) return (checkpointBaseFileName model)
  let pathModel = B8.pack $ basePath ++ "/" ++ modelName
      pathTrain = B8.pack $ basePath ++ "/" ++ trainName
  let inRef = getRef (inputLayerName $ tensorflowModel model)
      labRef = getRef (labelLayerName $ tensorflowModel model)
  let inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab
  let resetLastIO mdl = mdl {lastInputOutputTuple = Just (last inp, last lab)}
  let tf' = tensorflowModel model
  unless (null $ neuralNetworkVariables tf') $ TF.save pathModel (neuralNetworkVariables tf') >>= TF.run_
  unless (null $ trainingVariables tf') $
    TF.save pathTrain (trainingVariables tf' ++ concatMap optimizerRefsList (optimizerVariables tf')) >>= TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT]
  return $
    if isJust (checkpointBaseFileName model)
      then resetLastIO model
      else resetLastIO $ model {checkpointBaseFileName = Just basePath}

-- | Restore the model with the last input/output values.
restoreModelWithLastIO :: (MonadIO m) => TensorflowModel' -> SessionT m ()
restoreModelWithLastIO model =
  case lastInputOutputTuple model of
    Nothing     -> error "No last IO in restoreModelWithLastIO"
    Just (i, o) -> restoreModel model [i] [o]

-- | Build model (creates needed nodes in the Tensorflow session).
buildTensorflowModel :: TensorflowModel' -> SessionT IO ()
buildTensorflowModel tfModel = void $ tensorflowModelBuilder tfModel

-- | Restore a tensorflow model from the saved files.
restoreModel :: (MonadIO m) => TensorflowModel' -> Inputs -> Labels -> SessionT m ()
restoreModel tfModel inp lab = do
  basePath <- maybe (error "cannot restore from unknown location: checkpointBaseFileName is Nothing") return (checkpointBaseFileName tfModel)
  let pathModel = B8.pack $ basePath ++ "/" ++ modelName
      pathTrain = B8.pack $ basePath ++ "/" ++ trainName
  let inRef = getRef (inputLayerName $ tensorflowModel tfModel)
      labRef = getRef (labelLayerName $ tensorflowModel tfModel)
  let inpT = encodeInputBatch inp
      labT = encodeLabelBatch lab
  let tf' = tensorflowModel tfModel
  unless (null $ trainingVariables tf') $
    mapM (TF.restore pathTrain) (trainingVariables tf' ++ concatMap optimizerRefsList (optimizerVariables tf')) >>= TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT]
  unless (null $ neuralNetworkVariables tf') $ mapM (TF.restore pathModel) (neuralNetworkVariables tf') >>= TF.run_


instance Serialize TensorflowModel' where
  put tf@(TensorflowModel' (TensorflowModel inp out label train nnVars trVars optRefs) _ lastIO builder) = do
    put inp >> put out >> put label >> put (getTensorControlNodeName train) >> put lastIO
    put $ map getTensorRefNodeName nnVars
    put $ map getTensorRefNodeName trVars
    put optRefs
    let (mBasePath, bytesModel, bytesTrain) =
          unsafePerformIO $ do
            void $ runSession $ saveModelWithLastIO tf
            let basePath = fromMaybe (error "cannot read tensorflow model") (checkpointBaseFileName tf)
                pathModel = basePath ++ "/" ++ modelName
                pathTrain = basePath ++ "/" ++ trainName
            bModel <- liftIO $ BS.readFile pathModel
            bTrain <- liftIO $ BS.readFile pathTrain
            return (checkpointBaseFileName tf, bModel, bTrain)
    put mBasePath
    put bytesModel
    put bytesTrain
  get = do
    inp <- get
    out <- get
    label <- get
    train <- getControlNodeTensorFromName <$> get
    lastIO <- get
    nnVars <- map getRefTensorFromName <$> get
    trVars <- map getRefTensorFromName <$> get
    optRefs <- get
    mBasePath <- get
    bytesModel <- get
    bytesTrain <- get
    return $ force $
      unsafePerformIO $ do
        basePath <- maybe (getCanonicalTemporaryDirectory >>= flip createTempDirectory "") (\b -> createDirectoryIfMissing True b >> return b) mBasePath
        let pathModel = basePath ++ "/" ++ modelName
            pathTrain = basePath ++ "/" ++ trainName
        BS.writeFile pathModel bytesModel
        BS.writeFile pathTrain bytesTrain
        let fakeBuilder = TF.runSession $ return $ TensorflowModel inp out label train nnVars trVars optRefs
        return $ TensorflowModel' (TensorflowModel inp out label train nnVars trVars optRefs) (Just basePath) lastIO fakeBuilder

