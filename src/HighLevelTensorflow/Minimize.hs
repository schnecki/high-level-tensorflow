{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}

module HighLevelTensorflow.Minimize
    ( MinimizerRefs
    , minimizeWithRefs
    , gradientDescentRefs
    , adamRefs
    , adamRefs'
    ) where

import           Control.Monad          (zipWithM)
import           Data.Default           (Default (..))
import           Data.List              (zipWith4)


import qualified TensorFlow.Build       as TF (Build, explicitName)
import qualified TensorFlow.Core        as TF
import qualified TensorFlow.GenOps.Core as TFO (applyAdam, assignAdd)
import qualified TensorFlow.Gradient    as TF
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF hiding (assign, initializedVariable)
import qualified TensorFlow.Ops         as TFO (assign, initializedVariable,
                                                initializedVariable',
                                                zeroInitializedVariable)
import qualified TensorFlow.Output      as TF (OpDef (..))
import qualified TensorFlow.Tensor      as TF (Tensor (..), TensorKind, toBuild)

-- | Functions that minimize a loss w.r.t. a set of 'TF.Ref's.
--
-- Generally only performs one step of an iterative algorithm.
--
-- 'Minimizer's are defined as a function of the gradients instead of
-- the loss so that users can apply transformations to the gradients.
type MinimizerRefs a
   = forall m. TF.MonadBuild m =>
                 [TF.Tensor TF.Ref a] -> [TF.Shape] -> [TF.Tensor TF.Value a] -> m (TF.ControlNode, [TF.Tensor TF.Ref a], [TF.Tensor TF.Ref a])

-- | Convenience wrapper around 'TF.gradients' and a 'Minimizer'.
minimizeWithRefs :: (TF.MonadBuild m, TF.GradientCompatible a)
             => MinimizerRefs a
             -> TF.Tensor v a        -- ^ Loss.
             -> [TF.Tensor TF.Ref a] -- ^ Parameters of the loss function.
             -> [TF.Shape]           -- ^ Parameters shapes of the loss function.
             -> m (TF.ControlNode, [TF.Tensor TF.Ref a], [TF.Tensor TF.Ref a])
minimizeWithRefs minimizer loss params shapes = do
    let vals = map TF.value params
    TF.gradients loss vals >>= minimizer params shapes


-- | Perform one step of the gradient descent algorithm.
gradientDescentRefs :: TF.GradientCompatible a
                => a  -- ^ Learning rate.
                -> MinimizerRefs a
gradientDescentRefs learningRate params _ grads =
  TF.withNameScope "gradientDescent" $ do
    lrRef <- TFO.initializedVariable' (\x -> x {TF._opName = TF.explicitName "learningRate"}) (TF.scalar learningRate)
    let applyGrad param grad = TFO.assignAdd param (TF.neg lrRef `TF.mul` grad)
    gr <- TF.group =<< zipWithM applyGrad params grads
    return (gr, [], [lrRef])

-- | Perform one step of the adam algorithm.
--
-- See https://arxiv.org/abs/1412.6980.
--
-- NOTE: Currently requires all 'TF.Ref's to have an 'TF.initializedValue'.
adamRefs :: MinimizerRefs Float
adamRefs = adamRefs' def


toBuildTensor :: TF.TensorKind v => TF.Tensor v a -> TF.Tensor TF.Build a
toBuildTensor (TF.Tensor o) = TF.Tensor $ TF.toBuild o


adamRefs' :: TF.AdamConfig -> MinimizerRefs Float
adamRefs' config params shapes grads =
  TF.withNameScope "adamRefs" $ do
    lrRef <- TFO.initializedVariable' (\x -> x {TF._opName = TF.explicitName "learningRate"}) (TF.scalar $ TF.adamLearningRate config)
    let lr = toBuildTensor lrRef
    let beta1 = TF.scalar (TF.adamBeta1 config)
        beta2 = TF.scalar (TF.adamBeta2 config)
        epsilon = TF.scalar (TF.adamEpsilon config)
    -- Create adam state variables.
    ms <- mapM TFO.zeroInitializedVariable shapes
    vs <- mapM TFO.zeroInitializedVariable shapes
    beta1Power <- TFO.initializedVariable beta1
    beta2Power <- TFO.initializedVariable beta2
    -- Perform adam update.
    let applyGrad param m v = TFO.applyAdam param m v beta1Power beta2Power lr beta1 beta2 epsilon
    updateVars <- sequence $ zipWith4 applyGrad params ms vs grads
    -- Update beta variables after adam update.
    let updateBeta betaPower beta = TF.withControlDependencies updateVars (TFO.assign betaPower (betaPower `TF.mul` beta))
    updateBeta1 <- updateBeta beta1Power beta1
    updateBeta2 <- updateBeta beta2Power beta2
    grp <- TF.group (updateBeta1 : updateBeta2 : updateVars)
    let vars = ms ++ vs ++ [beta1Power, beta2Power]
    return (grp, vars, [lrRef])
