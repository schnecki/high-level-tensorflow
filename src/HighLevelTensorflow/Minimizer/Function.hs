{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}

module HighLevelTensorflow.Minimizer.Function
  ( gradientsWithRefs
  , minimizeWithRefs
  , minimizeGradientsWithRefs
  ) where


import qualified TensorFlow.Core                    as TF
import qualified TensorFlow.Gradient                as TF
import qualified TensorFlow.Tensor                  as TF (Tensor (..))


import           HighLevelTensorflow.Minimizer.Type

-- | Accumulate the gradients using the specified loss and loss function.
gradientsWithRefs ::
     (TF.MonadBuild m, TF.GradientCompatible a)
  => TF.Tensor v a        -- ^ Loss.
  -> [TF.Tensor TF.Ref a] -- ^ Parameters of the loss function.
  -> m [Gradients a]
gradientsWithRefs loss params = TF.gradients loss (map TF.value params)


-- | Simply applies the minimizer to the specified parameters. Could be used directly also!
minimizeGradientsWithRefs ::
     (TF.MonadBuild m)
  => MinimizerRefs a
  -> [TF.Tensor TF.Ref a]   -- ^ Parameters of the loss function.
  -> [TF.Shape]             -- ^ Parameters shapes of the loss function.
  -> [Gradients a]          -- ^ Gradients
  -> m (TF.ControlNode, [TF.Tensor TF.Ref a], [TF.Tensor TF.Ref a])
minimizeGradientsWithRefs minimizer = minimizer


-- | Convenience wrapper around 'TF.gradients' and a 'Minimizer'. Calculates the gradients and applies
minimizeWithRefs :: (TF.MonadBuild m, TF.GradientCompatible a)
             => MinimizerRefs a
             -> TF.Tensor v a        -- ^ Loss.
             -> [TF.Tensor TF.Ref a] -- ^ Parameters of the loss function.
             -> [TF.Shape]           -- ^ Parameters shapes of the loss function.
             -> m (TF.ControlNode, [TF.Tensor TF.Ref a], [TF.Tensor TF.Ref a])
minimizeWithRefs minimizer loss params shapes = do
    let vals = map TF.value params
    TF.gradients loss vals >>= minimizer params shapes


