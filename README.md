# High Level Tensorflow API

This is a high-level API to Tensorflow. It is used as a more convenient way to access the Tensorflow
Haskell API (https://github.com/tensorflow/haskell#readme and
https://tensorflow.github.io/haskell/haddock/doc-index-All.html).

*Current targeted Tensorflow Version: 1.14.0*

You need to add following to `stack.yaml`, as Tensorflow is not on stackage.

    - git: https://github.com/tensorflow/haskell
      commit: 8cde4d6a277f188ca495d29096a62b36e8534a3f
      subdirs:
      - tensorflow
      - tensorflow-core-ops
      - tensorflow-mnist
      - tensorflow-mnist-input-data
      - tensorflow-opgen
      - tensorflow-ops
      - tensorflow-proto


Tested with lts-14.25.

# Support

 - FeedForward ANNs
 - Minimizers:
   - Adam
   - RMSProp
   - SGD

Help in adding support for other algorithms and ANN types is very much appreciated.


# Example


    import qualified HighLevelTensorflow      as TF
    import           GHC.Int                  (Int64)
    import qualified Data.Vector.Storable                   as V

    modelBuilder :: ModelBuilderFunction    -- This is: (MonadBuild m) => Int64 -> m TensorflowModel
    modelBuilder colOut =
          buildModel $
          inputLayer1D inpLen >>
          fullyConnected [20] relu' >>
          fullyConnected [10] relu' >>
          fullyConnected [10] relu' >>
          fullyConnected [1, colOut] tanh' >>
          trainingByAdamWith AdamConfig {adamLearningRate = 0.001, adamBeta1 = 0.9, adamBeta2 = 0.999, adamEpsilon = 1e-8}
          -- trainingByGradientDescent 0.01
          -- trainingByRmsProp
          where inpLen = 10

then for instance use it with

    test :: IO ()
    test = do
      res <- TF.runSession $ do
        model <- modelBuilder 1
        let model' = TensorflowModel' model Nothing Nothing (modelBuilder 1)
        forwardRun model' [V.fromList [0.1,0.2 .. 1.0]]
      Prelude.print res


See the submodules of `HighLevelTensorflow.TensorflowModel` for more operations, like `backwardRun`,
`saveModel` and `restoreModel`.
