{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}

module HighLevelTensorflow.Minimizer.RMSProp
    ( RmsPropConfig (..)
    , rmsPropRefs
    , rmsPropRefs'
    ) where

import           Data.Default                       (Default (..))
import           Data.List                          (zipWith4)


import qualified TensorFlow.Build                   as TF (explicitName)
import qualified TensorFlow.Core                    as TF
import qualified TensorFlow.GenOps.Core             as TFO (applyRMSProp)
import qualified TensorFlow.Ops                     as TF hiding (assign,
                                                           initializedVariable)
import qualified TensorFlow.Ops                     as TFO (initializedVariable',
                                                            zeroInitializedVariable)
import qualified TensorFlow.Output                  as TF (OpDef (..))


import           HighLevelTensorflow.Minimizer.Type

-- | RMS Prop Configuration.
data RmsPropConfig = RmsPropConfig
    { rmsPropLearningRate :: !Float -- ^ Learning rate [Default: 0.001]
    , rmsPropRho          :: !Float -- ^ Decay rate [Default: 0.9]
    , rmsPropMomentum     :: !Float -- ^ Momentum [Default 0.0]
    , rmsPropEpsilon      :: !Float -- ^ Ridge Term [Default: 1e-7]
    }

instance Default RmsPropConfig where
  def = RmsPropConfig 0.001 0.9 0.0 1e-7


-- | Perform one step of RMSProp.
rmsPropRefs :: MinimizerRefs Float
rmsPropRefs = rmsPropRefs' def

-- | RMS Prop.
rmsPropRefs' :: RmsPropConfig -> MinimizerRefs Float
rmsPropRefs' config params shapes grads =
  TF.withNameScope "rmsPropRefs" $ do
    lrRef <- TFO.initializedVariable' (\x -> x {TF._opName = TF.explicitName "learningRate"}) (TF.scalar $ rmsPropLearningRate config)
    let lr = toBuildTensor lrRef
    let rho = TF.scalar (rmsPropRho config)
        momentum = TF.scalar (rmsPropMomentum config)
        epsilon = TF.scalar (rmsPropEpsilon config)
    -- Create rmsProp state variables.
    ms <- mapM TFO.zeroInitializedVariable shapes
    moms <- mapM TFO.zeroInitializedVariable shapes
    -- -- Perform rmsProp update.
    let applyGrad param m mom = TFO.applyRMSProp param m mom lr rho momentum epsilon
    updateVars <- sequence $ zipWith4 applyGrad params ms moms grads
    -- Make a contorl node out of the result and return the value.
    grp <- TF.group updateVars
    let vars = ms ++ moms
    return (grp, vars, [lrRef])
