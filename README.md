# Distributed LR with stochastic-gradient-decent
Implementation of parallel Stochastic gradient decent based on *Zinkevich, M., Weimer, M., Li, L., &amp; Smola, A. J. (2010). Parallelized stochastic gradient descent, In Advances in neural information processing systems (pp. 2595-2603)*

# Dataset- 
* Dynamic Features of VirusShare Executables Data Set https://archive.ics.uci.edu/ml/datasets/Dynamic+Features+of+VirusShare+Executables

# Dependencies - 
* Scikit-learn
* Mpi4py

# Linear regression -
![alt text](1.png)

# Parallel Stocastic Gradient decent -
![alt text](2.png)


##### Fig 1
![alt text](3.png)

##### Fig 2
![alt text](4.png)


# Observations and results : 
**Preprocessing  :**  
I read the data line wise from each file and stored it into an array object. The code is available in the .py files attached (in the beginning). Both predictors and targets are stored into an array. Train size = 70 percent. Rest is test data to use for RMSE.

**Performance and convergence of PSGD for Dynamic features dataset :**
The program can run with any number of processes. The data segmentation used in the code can handle all data sizes even when size not divisible by number of processes. 

To run the file use the following command from the terminal :
```python
mpiexec -n numprocs python -m mpi4py filename.py
```
![alt text](5.png)


![alt text](6.png)



