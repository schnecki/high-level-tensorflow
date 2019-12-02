module HighLevelTensorflow.Util
    ( getTensorControlNodeName
    , getTensorRefNodeName
    , getRefTensorFromName
    , getControlNodeTensorFromName
    , getRef
    , replace
    ) where

import           Data.Serialize.Text ()
import           Data.Text           (Text)

import qualified TensorFlow.Core     as TF
import qualified TensorFlow.Output   as TF (ControlNode (..), NodeName (..))
import qualified TensorFlow.Tensor   as TF (Tensor (..), tensorNodeName, tensorRefFromName)


getTensorControlNodeName :: TF.ControlNode -> Text
getTensorControlNodeName = TF.unNodeName . TF.unControlNode

getTensorRefNodeName :: TF.Tensor TF.Ref a -> Text
getTensorRefNodeName = TF.unNodeName . TF.tensorNodeName

getRefTensorFromName :: Text -> TF.Tensor TF.Ref a
getRefTensorFromName = TF.tensorRefFromName

getControlNodeTensorFromName :: Text -> TF.ControlNode
getControlNodeTensorFromName = TF.ControlNode . TF.NodeName

getRef :: Text -> TF.Tensor TF.Ref Float
getRef = TF.tensorFromName


replace :: Int -> a -> [a] -> [a]
replace idx val ls = take idx ls ++ val : drop (idx+1) ls
