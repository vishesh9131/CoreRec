def fibonacci(n):
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib

n = 10  # Change this to generate Fibonacci sequence up to a different number
print(fibonacci(n))
