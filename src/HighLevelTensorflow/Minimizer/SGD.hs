{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}

module HighLevelTensorflow.Minimizer.SGD
  ( gradientDescentRefs
  ) where

import           Control.Monad                      (zipWithM)


import qualified TensorFlow.Build                   as TF (explicitName)
import qualified TensorFlow.Core                    as TF
import qualified TensorFlow.GenOps.Core             as TFO (assignAdd)
import qualified TensorFlow.Gradient                as TF
import qualified TensorFlow.Ops                     as TF hiding (assign,
                                                           initializedVariable)
import qualified TensorFlow.Ops                     as TFO (initializedVariable')
import qualified TensorFlow.Output                  as TF (OpDef (..))


import           HighLevelTensorflow.Minimizer.Type

-- | Perform one step of the gradient descent algorithm.
gradientDescentRefs ::
     TF.GradientCompatible a
  => a -- ^ Learning rate.
  -> MinimizerRefs a
gradientDescentRefs learningRate params _ grads =
  TF.withNameScope "gradientDescent" $ do
    lrRef <- TFO.initializedVariable' (\x -> x {TF._opName = TF.explicitName "learningRate"}) (TF.scalar learningRate)
    let applyGrad param grad = TFO.assignAdd param (TF.neg lrRef `TF.mul` grad)
    gr <- TF.group =<< zipWithM applyGrad params grads
    return (gr, [], [lrRef])

