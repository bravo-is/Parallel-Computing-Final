# Parallel-Computing-Final
## Conway's Game of Life implemented in CUDA

### Files
GameOfLife.cu - Will start with the same board state and produce the same patterns every run.  ![GOL](https://raw.githubusercontent.com/bravo-is/Parallel-Computing-Final/master/GOL.png)
RandomGameOfLife.cu - Populates the board randomly leading to new patterns every compilation.  ![RandomGOL](https://raw.githubusercontent.com/bravo-is/Parallel-Computing-Final/master/RandomGOL.png)
### Execution  
To make the execution of this project simple we used some existing libraries from CUDA BY EXAMPLE. Available from https://developer.nvidia.com/cuda-example (source code **.zip**)

1. Go to the 'CUDA by Example' source code directory and create a new folder.
![FileExplorer](https://raw.githubusercontent.com/bravo-is/Parallel-Computing-Final/master/FileExplorer.png)
2. Place all the files from this repo into the new folder.
3. Compile the .cu files with nvcc.
4. Run the executables.
