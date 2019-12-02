{-# LANGUAGE BangPatterns #-}
module HighLevelTensorflow.TensorflowModel.Type
    ( TensorflowModel (..)
    , TensorflowModel' (..)

    ) where

import           Control.DeepSeq
import           Data.Serialize.Text                    ()
import           Data.Text                              (Text)

import qualified TensorFlow.Core                        as TF
import qualified TensorFlow.Output                      as TF (ControlNode (..))
import qualified TensorFlow.Tensor                      as TF (Tensor (..))

import           HighLevelTensorflow.OptimizerVariables


-- | A Tensorflow Model holds all needed information to work with a model in RAM.
data TensorflowModel = TensorflowModel
  { inputLayerName         :: Text                     -- ^ Input layer name for feeding input.
  , outputLayerName        :: Text                     -- ^ Output layer name for predictions.
  , labelLayerName         :: Text                     -- ^ Labels input layer name for training.
  , trainingNode           :: TF.ControlNode           -- ^ Training node.
  , neuralNetworkVariables :: [TF.Tensor TF.Ref Float] -- ^ Neural network variables for saving and restoring.
  , trainingVariables      :: [TF.Tensor TF.Ref Float] -- ^ Training data/settings for saving and restoring.
  , optimizerVariables     :: [OptimizerVariables]
  }

instance NFData TensorflowModel where
  rnf (TensorflowModel i o l !_ !_ !_ !_) = rnf i `seq` rnf o `seq` rnf l


-- | A @TensorflowModel'@ holds the information how to load and store and underlying @TensorflowModel@, as well as how
-- to build the model (required for loading and starting a new Tensorflow session).
data TensorflowModel' = TensorflowModel'
  { tensorflowModel        :: TensorflowModel
  , checkpointBaseFileName :: Maybe FilePath
  , lastInputOutputTuple   :: Maybe ([Float], [Float])
  , tensorflowModelBuilder :: TF.Session TensorflowModel
  }


instance NFData TensorflowModel' where
  rnf (TensorflowModel' m f l !_) = rnf m `seq` rnf f `seq` rnf l

