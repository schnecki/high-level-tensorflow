{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
module HighLevelTensorflow.TensorflowModel.Ops
    ( setLearningRates
    , getLearningRates
    , forwardRun
    , backwardRun
    , computeGradients
    , copyValuesFromTo
    , saveModel
    , saveModelWithLastIO
    , restoreModel
    , restoreModelWithLastIO
    , buildTensorflowModel
    ) where

import           Control.Arrow                            ((***))
import           Control.DeepSeq
import           Control.Monad                            (unless, void, zipWithM)
import           Control.Monad.IO.Class                   (MonadIO, liftIO)
import qualified Data.ByteString                          as BS
import qualified Data.ByteString.Char8                    as B8
import           Data.List                                (genericLength)
import           Data.Maybe                               (fromMaybe, isJust)
import           Data.Serialize
import           Data.Serialize.Text                      ()
import qualified Data.Vector                              as VU
import qualified Data.Vector.Storable                     as V
import           System.Directory
import           System.IO.Temp
import           System.IO.Unsafe

import qualified TensorFlow.Core                          as TF
import qualified TensorFlow.GenOps.Core                   as TF (print, square, sub)
import qualified TensorFlow.Gradient                      as TF

import qualified TensorFlow.Ops                           as TF hiding
                                                                 (initializedVariable,
                                                                 zeroInitializedVariable)

import           TensorFlow.Session

import           HighLevelTensorflow.OptimizerVariables
import           HighLevelTensorflow.TensorflowModel.Type
import           HighLevelTensorflow.Util

type Outputs = [V.Vector Float]        -- ^ 1st level: number of input rows, 2nd level: number of actions
type Inputs = [V.Vector Float]
type Labels = [V.Vector Float]

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
    lrRefs = concatMap getLearningRateRef (optimizerVariables $ tensorflowModel model)

-- | Get all learning rates of all optimizers.
getLearningRates :: (MonadIO m) => TensorflowModel' -> SessionT m [Float]
getLearningRates model = do
  lrValues <- TF.run lrRefs
  return $ map VU.head (lrValues :: [VU.Vector Float])
  where
    lrRefs = concatMap getLearningRateRef (optimizerVariables $ tensorflowModel model)

-- | This helper function encodes the input batch.
encodeInputBatch :: Inputs -> TF.TensorData Float
encodeInputBatch xs = TF.encodeTensorData [genericLength xs, fromIntegral $ V.length (head' xs)] (mconcat xs)
  where
    head' []    = error "head: empty input data in encodeInputBatch"
    head' (v:_) = v

encodeLabelBatch :: Labels -> TF.TensorData Float
encodeLabelBatch xs = TF.encodeTensorData [genericLength xs, fromIntegral $ V.length (head' xs)] (mconcat xs)
  where head' []    = error "head: empty input data in encodeLabelBatch"
        head' (x:_) = x

-- | Forward run the model.
forwardRun :: (MonadIO m) => TensorflowModel' -> Inputs -> SessionT m Outputs
forwardRun model inp =
  let inRef = getRef (inputLayerName $ tensorflowModel model)
      outRef = getRef (outputLayerName $ tensorflowModel model)
      inpT = encodeInputBatch inp
      nrOuts = length inp
   in do (res :: V.Vector Float) <- TF.runWithFeeds [TF.feed inRef inpT] outRef
         return $ separateInputRows 0 (V.length res `div` nrOuts) res []

separateInputRows :: (Show a, V.Storable a) => Int -> Int -> V.Vector a -> [V.Vector a] -> [V.Vector a]
separateInputRows i len vec acc
  | V.length vec == i = reverse acc
  | V.length vec < i = error $ "error in separate (in Tensorflow.forwardRun), number of values did not match: " ++ show vec ++ " - len: " ++ show len
  | otherwise = separateInputRows (i + len) len vec (V.slice i len vec : acc)

-- | Train tensorflow model with checks.
backwardRun :: (MonadIO m) => TensorflowModel' -> Inputs -> Labels -> SessionT m ()
backwardRun model inp lab
  | null inp || any V.null inp || null lab = error $ "Empty parameters in backwardRun not allowed! inp: " ++ show inp ++ ", lab: " ++ show lab
  | otherwise =
    let inRef = getRef (inputLayerName $ tensorflowModel model)
        labRef = getRef (labelLayerName $ tensorflowModel model)
        inpT = encodeInputBatch inp
        labT = encodeLabelBatch $ map (V.map (max (-trainMaxVal) . min trainMaxVal)) lab
     in TF.runWithFeeds_ [TF.feed inRef inpT, TF.feed labRef labT] (trainingNode $ tensorflowModel model)

computeGradients :: (MonadIO m) => TensorflowModel' -> Inputs -> Labels -> SessionT m [V.Vector Float]
computeGradients model inp lab
  | null inp || any V.null inp || null lab = error $ "Empty parameters in computeGradients not allowed! inp: " ++ show inp ++ ", lab: " ++ show lab
  | otherwise =
    let inRef = getRef (inputLayerName $ tensorflowModel model)
        labRef = getRef (labelLayerName $ tensorflowModel model)
        grads = getRefTensorFromName "gradients"
        inpT = encodeInputBatch inp
        labT = encodeLabelBatch $ map (V.map (max (-trainMaxVal) . min trainMaxVal)) lab
        nrOuts = length inp
     in do (res :: V.Vector Float) <- TF.runWithFeeds [TF.feed inRef inpT, TF.feed labRef labT] grads
           liftIO $ print res
           return $ separateInputRows 0 (V.length res `div` nrOuts) res []
      -- res <- TF.runWithFeeds [TF.feed inRef inpT, TF.feed labRef labT] grads
      -- liftIO $ print res


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
  put tf@(TensorflowModel' (TensorflowModel inp out label train nnVars trVars optRefs) _ mLastIO builder) = do
    put inp >> put out >> put label >> put (getTensorControlNodeName train) >> put (fmap (V.toList *** V.toList) mLastIO)
    put $ map getTensorRefNodeName nnVars
    put $ map getTensorRefNodeName trVars
    put optRefs
    let (mBasePath, bytesModel, bytesTrain) =
          unsafePerformIO $
            -- void $ saveModelWithLastIO tf -- must have been done before
           do
            let basePath = fromMaybe (error "cannot read tensorflow model") (checkpointBaseFileName tf) -- models have been saved during conversion
                pathModel = basePath ++ "/" ++ modelName
                pathTrain = basePath ++ "/" ++ trainName
            bModel <- liftIO $ BS.readFile pathModel
            bTrain <- liftIO $ BS.readFile pathTrain
            removeFile pathModel -- remove file to not litter the filesystem
            removeFile pathTrain
            return (checkpointBaseFileName tf, bModel, bTrain)
    put mBasePath
    put bytesModel
    put bytesTrain
  get = do
    inp <- get
    out <- get
    label <- get
    train <- getControlNodeTensorFromName <$> get
    lastIO <- fmap (V.fromList *** V.fromList) <$> get
    nnVars <- map getRefTensorFromName <$> get
    trVars <- map getRefTensorFromName <$> get
    optRefs <- get
    mBasePath <- get
    bytesModel <- get
    bytesTrain <- get
    return $
      force $
      unsafePerformIO $ do
        basePath <- maybe (getCanonicalTemporaryDirectory >>= flip createTempDirectory "") (\b -> createDirectoryIfMissing True b >> return b) mBasePath
        let pathModel = basePath ++ "/" ++ modelName
            pathTrain = basePath ++ "/" ++ trainName
        BS.writeFile pathModel bytesModel
        BS.writeFile pathTrain bytesTrain
        let fakeBuilder = TF.runSession $ return $ TensorflowModel inp out label train nnVars trVars optRefs
        return $ TensorflowModel' (TensorflowModel inp out label train nnVars trVars optRefs) (Just basePath) lastIO fakeBuilder


