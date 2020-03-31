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
