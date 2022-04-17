# CNN_fused
## Introduction
This is a reopsitory which implements convolutional neurall network basic unit using only numpy in Python  
Also, this repository try to fuse batch normalization and convolution operation into one function to reduce the memory access

## Package required
For now, numpy is the only package you need in this project

## Running the code
Just run the file main.py, and you can see the output like this in the terminal:
```
Trial 15
Original time:  1.9979476928710938 # time cost of basic CNN
Fused time:  0.9942054748535156 # time cost of fused_CNN
	 X: (1, 43, 75, 2) # shapes generated randomly
	 y: (1, 43, 75, 2)
	 y_fused: (1, 43, 75, 2)
	 W: (3, 3, 2, 2)
	 b: (1, 1, 1, 2)
	 dLdW: (3, 3, 2, 2)
	 dLdB: (1, 1, 1, 2)
	 dLdX: (1, 43, 75, 2)

Process finished with exit code 0
```

Besides, you can also specify the shape of input by modifying the following code block:
```python
        n_ex = np.random.randint(1, 10)  # batch_size
        in_rows = np.random.randint(1, 224) # row size
        in_cols = np.random.randint(1, 224) # column size
        n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)  # input and output channel size
```

## Performance analysis
In each experiment, i generate random inputs with random shapes for 15 times, and the time fused_CNN faster than basic CNN is always greater than 10, which validates the better performance of fusion operation.  
The higher the data volume(here, we means the input shape) is, the more significant the performance superiority will be  

## Future work
1.In some cases, the fuse_CNN will be slower than the basic CNN, we still need to probe deeper into this issue.  
2.Due to my busy schedule, i haven't tried to build a backbone network using this fuse_CNN unit for more experiments  
3.Actually, for optimization, i also try to implement the optimization method proposed by this paper：  
[Tensorfolding: Improving Convolutional Neural Network Performance with Fused Microkernels]  
(https://sc18.supercomputing.org/proceedings/tech_poster/poster_files/post155s2-file3.pdf)  
but this approach can't work for me, the reason still need to be mined.  
4.Some parallel computing techniques and computation efficiency analysis still need to be adopted  
