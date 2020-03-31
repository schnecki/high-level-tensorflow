module HighLevelTensorflow
    ( module H
    , module TF
    ) where


import           HighLevelTensorflow.Minimizer          as H
import           HighLevelTensorflow.OptimizerVariables as H
import           HighLevelTensorflow.TensorflowModel    as H

import           TensorFlow.Core                        as TF
import           TensorFlow.GenOps.Core                 as TF hiding (noOp)
import           TensorFlow.Session                     as TF

-- import qualified TensorFlow.Build       as TF (addNewOp, evalBuildT, explicitName, opDef,
--                                                opDefWithName, opType, runBuildT, summaries)
-- import qualified TensorFlow.Core        as TF hiding (value)
-- import qualified TensorFlow.GenOps.Core as TF (abs, add, approximateEqual,
--                                                approximateEqual, assign, cast,
--                                                getSessionHandle, getSessionTensor,
--                                                identity', lessEqual, matMul, mul,
--                                                readerSerializeState, relu', shape, square,
--                                                sub, tanh, tanh', truncatedNormal)
-- import qualified TensorFlow.Minimize    as TF
-- import qualified TensorFlow.Ops         as TF (initializedVariable, initializedVariable',
--                                                placeholder, placeholder', reduceMean,
--                                                reduceSum, restore, save, scalar, vector,
--                                                zeroInitializedVariable,
--                                                zeroInitializedVariable')
-- import qualified TensorFlow.Tensor      as TF (Ref (..), collectAllSummaries,
--                                                tensorNodeName, tensorRefFromName,
--                                                tensorValueFromName)


-- modelBuilder :: ModelBuilderFunction    -- This is: (MonadBuild m) => Int64 -> m TensorflowModel
-- modelBuilder colOut =
--       buildModel $
--       inputLayer1D inpLen >>
--       fullyConnected [20] relu' >>
--       fullyConnected [10] relu' >>
--       fullyConnected [10] relu' >>
--       fullyConnected [1, colOut] tanh' >>
--       trainingByAdamWith AdamConfig {adamLearningRate = 0.001, adamBeta1 = 0.9, adamBeta2 = 0.999, adamEpsilon = 1e-8}
--       -- trainingByGradientDescent 0.01
--       -- trainingByRmsProp
--       where inpLen = 10

-- test :: IO ()
-- test = do
--   res <- TF.runSession $ do
--     model <- modelBuilder 1
--     let model' = TensorflowModel' model Nothing Nothing (modelBuilder 1)
--     forwardRun model' [V.fromList [0.1,0.2 .. 1.0]]
--   Prelude.print res
