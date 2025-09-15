In this directory are stored some examples

# InheritanceExample.ipynb
This Jupyter Notebook displays an example where a child class inherits BaseDLFramework to solve a classification case for the sklearn IRIS datast.

Since a validation and test set have also been constructed, the child class will include a validation algorithm and a test algorithm. The run_epochs method has been overridden to include the validation phase inside the loop
Furthermore the best model is now saved when the best validation metric is achieved

A test method has been included, along with more methods to plot data.

# MLPexample.py and trainer_batch.sh
**MLPexample.py** follows the same algorithm as the **InheritanceExample.py**. However, it is now initialized via argparse, in order to be directly run by bash.
Here's an example of a bash code:

<pre># After sourcing the Python environment
python3 MLPexample.py --hidden_dims "[16]" --lr 0.001 --epoch_max 20" </pre>

Notice that the --hidden_dims argument is a python list written as a string.

The batch script **trainer_batch.sh** is an example on how multiple programs with different arguments can be run. To run the script:

<pre>chmod +x trainer_batch.sh
./trainer_batch.sh </pre>

# MLPexampleDDP.py and DDP_batch.sh
**MLPexample.py** follows the same algorithm as the **InheritanceExample.py**. However, it is now initialized via argparse, in order to be directly run by bash.
Here's an example of a bash code, for running on a single node:

<pre># After sourcing the Python environment
torchrun --standalone --nproc_per_node=gpu MLPexampleDDP.py --hidden_dims "$hidden_dims" --lr $lr --epoch_max $epoch_max > "$logfile" 2>&1

Notice that the --hidden_dims argument is a python list written as a string.

The batch script **trainer_batch.sh** is an example on how multiple programs with different arguments can be run. To run the script:


<pre>chmod +x DDP_batch.sh
./DDP_batch.sh </pre>
