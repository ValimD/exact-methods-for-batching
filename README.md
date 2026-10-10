# Primal Dual

This repository contains an implementation of the Primal-Dual algorithm for linear optimization problems in standard form. To evaluate the algorithm, the Minimum Cut and Maximum Flow with Multiple Goods problems were used.

In addition to Primal-Dual, the repository includes PuLP, which was used to compare results and runtime. Specifically for the Minimum Cut problem, the Stoer-Wagner algorithm was also implemented, which allows one to find the global minimum cut and compare the results obtained by the different methods.

## Repository Structure

The repository is organized into two main directories: `datasets`, which contains the instances used in the tests, and `src`, which contains the code for the methods. The files in the project root are intended for environment configuration, with the exception of `exec.sh`, which runs all instances and saves the objective function and the Primal-Dual runtime in a CSV file.

The `src` directory contains the `main.py` file, which serves as the entry point for executing the methods, and the following subdirectories:

- `methods`: contains the implementations of the methods used.
- `problems`: contains the implementations of the problems, which are responsible for converting them to the parameters expected by the methods.

## Execution

To run the methods, first clone the repository:

```bash
git clone https://github.com/ValimD/primal-dual
cd primal-dual
```

If you are using the `mise` tool, configure the runtime environment with the following commands:

```bash
mise trust
mise install
```

After setting up the environment, the methods can be executed using the following command:

```bash
python src/main.py <problem> <instance_path> [optional flags]
```

The `problem` argument is required and can be a value between 0 and 2. A value of 0 will run the Primal-Dual algorithm for two simple problems; a value of 1 will run the Primal-Dual, PuLP, and Stoer-Wagner algorithms for the Min Cut problem; and finally, a value of 2 will do the same as the previous one (except for Stoer) for the Maximum Flow problem.

The `instance_path` argument is required only if the previous argument is 1 or 2. Finally, the optional flags are `--max-iter`, which specifies the maximum number of iterations, and `--prizes`, which specifies the weights for the items in the maximum flow problem (the default is 1 for all).

## Results

The test results are presented in the table below, showing the objective function values (Z) and the execution times (T) for each method.

| Instância | Z Primal-Dual | T Primal-Dual | Z PuLP | T PuLP | Z Stoer-Wagner | T Stoer-Wagner |
| --------- | ------------- | ------------- | ------ | ------ | -------------- | -------------- |
| instance1.min | 18.0 | 0.001980s | 18.0 | 0.009265s | 13.0 | 0.000126s |
| instance2.min | 46.0 | 0.007854s | 46.0 | 0.009696s | 20.0 | 0.000854s |
| instance3.min | 129.0 | 0.251284s | 129.0 | 0.012683s | 31.0 | 0.005825s |
| instance4.min | 65.0 | 2.675358s | 65.0 | 0.022349s | 65.0 | 0.018757s |
| instance5.min | 198.0 | 3.742032s | 198.0 | 0.019705s | 43.0 | 0.045566s |
| mc_instance1.max | 20.0 | 0.001551s | 20.0 | 0.009392s | ... | ... |
| mc_instance2.max | 58.0 | 0.140142s | 58.0 | 0.010668s | ... | ... |
| mc_instance3.max | 168.0 | 7.644417s | 168.0 | 0.024094s | ... | ... |
| mc_instance4.max | 307.0 | 24.546699s | 307.0 | 0.043370s | ... | ... |
| mc_instance5.max | 138.0 | 448.0 | 138.0 | 0.043524s | ... | ... |