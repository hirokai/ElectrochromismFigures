{-# LANGUAGE TypeFamilies #-}

class Task a where
    type Output a :: *
    run :: a -> IO ()

runSeq :: (Task a, Task b) => a -> b -> CompositeTask
runSeq

data CompositeTask = CompositeTask
data Correction = Correction String

instance Task Correction where
    type Output Correction = [String]
    run t = do
        putStrLn "Stub"


main = do
    run (Correction "hoge")

