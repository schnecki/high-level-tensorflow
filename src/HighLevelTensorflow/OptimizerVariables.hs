{-# LANGUAGE BangPatterns #-}
module HighLevelTensorflow.OptimizerVariables
    ( OptimizerVariables (..)
    , prettyOptimizerNames
    , optimizerRefsList
    , getLearningRateRefs
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
      { gradientDescentLearningRateRef :: TF.Tensor TF.Ref Float
      }
  | AdamRefs
      { adamLearningRateRef :: TF.Tensor TF.Ref Float
      }

instance Serialize OptimizerVariables where
  put (GradientDescentRefs lr) = put (0 :: Int) >> put (getTensorRefNodeName lr)
  put (AdamRefs lr)            = put (1 :: Int) >> put (getTensorRefNodeName lr)
  get = do
    nr <- get
    case (nr :: Int) of
      0 -> do
        lr <- getRefTensorFromName <$> get
        return $ GradientDescentRefs lr
      1 -> do
        lr <- getRefTensorFromName <$> get
        return $ AdamRefs lr
      x -> error $ "Could not deserialise optimizer refs with key: " <> show x

instance NFData OptimizerVariables where
  rnf (GradientDescentRefs !_) = ()
  rnf (AdamRefs !_ )           = ()

prettyOptimizerNames :: OptimizerVariables -> String
prettyOptimizerNames GradientDescentRefs{} = "Gradient Descent"
prettyOptimizerNames AdamRefs{}            = "Adam"


optimizerRefsList :: OptimizerVariables -> [TF.Tensor TF.Ref Float]
optimizerRefsList (GradientDescentRefs lr) = [lr]
optimizerRefsList (AdamRefs lr)            = [lr]

getLearningRateRefs :: OptimizerVariables -> [TF.Tensor TF.Ref Float]
getLearningRateRefs (GradientDescentRefs lr) = [lr]
getLearningRateRefs (AdamRefs lr)            = [lr]
