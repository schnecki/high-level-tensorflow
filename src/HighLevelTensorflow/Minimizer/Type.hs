{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}

module HighLevelTensorflow.Minimizer.Type
  ( MinimizerRefs
  , Gradients
  , toBuildTensor
  ) where


import qualified TensorFlow.Core   as TF
import qualified TensorFlow.Tensor as TF (Tensor (..), TensorKind, toBuild)


-- | Functions that minimize a loss w.r.t. a set of 'TF.Ref's.
--
-- Generally only performs one step of an iterative algorithm.
--
-- 'Minimizer's are defined as a function of the gradients instead of
-- the loss so that users can apply transformations to the gradients.
type MinimizerRefs a
   = forall m. TF.MonadBuild m =>
                 [TF.Tensor TF.Ref a] -> [TF.Shape] -> [TF.Tensor TF.Value a] -> m (TF.ControlNode, [TF.Tensor TF.Ref a], [TF.Tensor TF.Ref a])


-- | Type for gradients for better readability.
type Gradients a = TF.Tensor TF.Value a


-- | Helper function to lift a tensor into a Build.
toBuildTensor :: TF.TensorKind v => TF.Tensor v a -> TF.Tensor TF.Build a
toBuildTensor (TF.Tensor o) = TF.Tensor $ TF.toBuild o

