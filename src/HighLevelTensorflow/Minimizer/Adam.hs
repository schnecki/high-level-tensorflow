{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}

module HighLevelTensorflow.Minimizer.Adam
  ( AdamConfig(..)
  , adam
  , adamRefs
  , adam'
  , adamRefs'
  ) where

import           Data.Default                       (Default (..))
import           Data.List                          (zipWith4)


import qualified TensorFlow.Build                   as TF (explicitName)
import qualified TensorFlow.Core                    as TF
import qualified TensorFlow.GenOps.Core             as TFO (applyAdam)
import qualified TensorFlow.Ops                     as TF hiding (assign,
                                                           initializedVariable)
import qualified TensorFlow.Ops                     as TFO (assign, initializedVariable,
                                                            initializedVariable',
                                                            zeroInitializedVariable)
import qualified TensorFlow.Output                  as TF (OpDef (..))


import           HighLevelTensorflow.Minimizer.Type

import           TensorFlow.Minimize                (AdamConfig (..), adam, adam')

-- | Perform one step of the adam algorithm. See `adam`.
--
-- See https://arxiv.org/abs/1412.6980.
--
-- NOTE: Currently requires all 'TF.Variable's to have an 'TF.initializedValue'.
adamRefs :: MinimizerRefs Float
adamRefs = adamRefs' def

-- | See `adam` and `AdamConfig`.
adamRefs' :: AdamConfig -> MinimizerRefs Float
adamRefs' config params shapes grads =
  TF.withNameScope "adamRefs" $ do
    lrRef <- TFO.initializedVariable' (\x -> x {TF._opName = TF.explicitName "learningRate"}) (TF.scalar $ adamLearningRate config)
    let lr = toBuildTensor lrRef
    let beta1 = TF.scalar (adamBeta1 config)
        beta2 = TF.scalar (adamBeta2 config)
        epsilon = TF.scalar (adamEpsilon config)
    -- Create adam state variables.
    ms <- mapM TFO.zeroInitializedVariable shapes
    vs <- mapM TFO.zeroInitializedVariable shapes
    beta1Power <- TFO.initializedVariable beta1
    beta2Power <- TFO.initializedVariable beta2
    -- Perform adam update.
    let applyGrad param m v = TFO.applyAdam param m v beta1Power beta2Power lr beta1 beta2 epsilon
    -- let applyGrad param m v = TF.resourceApplyAdam param m v (TF.readValue beta1Power) (TF.readValue beta2Power) lr beta1 beta2 epsilon
    updateVars <- sequence $ zipWith4 applyGrad params ms vs grads
    -- Update beta variables after adam update.
    let updateBeta betaPower beta = TF.withControlDependencies updateVars (TFO.assign betaPower (betaPower `TF.mul` beta))
    updateBeta1 <- updateBeta beta1Power beta1
    updateBeta2 <- updateBeta beta2Power beta2
    grp <- TF.group (updateBeta1 : updateBeta2 : updateVars)
    let vars = ms ++ vs ++ [beta1Power, beta2Power]
    return (grp, vars, [lrRef])
