{-# LANGUAGE BangPatterns #-}
module HighLevelTensorflow.OptimizerVariables
    ( OptimizerVariables (..)
    , prettyOptimizerNames
    , optimizerRefsList
    , getLearningRateRef
    ) where

import           Control.DeepSeq
import           Data.Serialize
import           Data.Serialize.Text      ()

import qualified TensorFlow.Core          as TF
import qualified TensorFlow.Tensor        as TF (Tensor (..))

import           HighLevelTensorflow.Util

-- | This data structure saves the type and references to the model variables of the optimizer.
data OptimizerVariables
  = GradientDescentRefs
      { gradientDescentLearningRateRef :: !(TF.Tensor TF.Ref Float)
      }
  | AdamRefs
      { adamLearningRateRef :: !(TF.Tensor TF.Ref Float)
      }
  | RmsPropRefs
      { rmsPropLearningRateRef :: !(TF.Tensor TF.Ref Float)
      }

instance NFData OptimizerVariables where
  rnf (GradientDescentRefs !_) = ()
  rnf (AdamRefs !_ )           = ()
  rnf (RmsPropRefs !_ )        = ()

instance Serialize OptimizerVariables where
  put (GradientDescentRefs lr) = put (0 :: Int) >> put (getTensorRefNodeName lr)
  put (AdamRefs lr)            = put (1 :: Int) >> put (getTensorRefNodeName lr)
  put (RmsPropRefs lr)         = put (2 :: Int) >> put (getTensorRefNodeName lr)
  get = do
    nr <- get
    case (nr :: Int) of
      0 -> do
        lr <- getRefTensorFromName <$> get
        return $ GradientDescentRefs lr
      1 -> do
        lr <- getRefTensorFromName <$> get
        return $ AdamRefs lr
      2 -> do
        lr <- getRefTensorFromName <$> get
        return $ RmsPropRefs lr
      x -> error $ "Could not deserialise optimizer refs with key: " <> show x


prettyOptimizerNames :: OptimizerVariables -> String
prettyOptimizerNames GradientDescentRefs{} = "Gradient Descent"
prettyOptimizerNames AdamRefs{}            = "Adam"
prettyOptimizerNames RmsPropRefs{}         = "RmsProp"


optimizerRefsList :: OptimizerVariables -> [TF.Tensor TF.Ref Float]
optimizerRefsList (GradientDescentRefs lr) = [lr]
optimizerRefsList (AdamRefs lr)            = [lr]
optimizerRefsList (RmsPropRefs lr)         = [lr]

getLearningRateRef :: OptimizerVariables -> [TF.Tensor TF.Ref Float]
getLearningRateRef (GradientDescentRefs lr) = [lr]
getLearningRateRef (AdamRefs lr)            = [lr]
getLearningRateRef (RmsPropRefs lr)         = [lr]

