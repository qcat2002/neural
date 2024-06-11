import regression
import time

def test():
    X, Y = regression.read_data()
    # regression.dispay(X, Y)
    dimension = X.shape[1]

    # initialise
    w = regression.np.array([0 for _ in range(dimension)])
    b = 0

    # non-vectorized linear model
    start_time = time.time()
    p1 = regression.linearmodel(X, w, b)
    end_time = time.time()
    timer1 = end_time - start_time

    # vectorized
    vec_X, vec_w = regression.vectorized(X, w, b)

    # vectorized linear model
    start_time = time.time()
    p2 = regression.vectorized_linearmodel(vec_X, vec_w)
    end_time = time.time()
    timer2 = end_time - start_time
    # print(timer1)
    # print(timer2)
    # print(timer2 < timer1)

    # cost function
    loss = regression.cost(p2, Y)

    # optimized by close form
    w_vec = regression.solve_exact(vec_X, Y)
    p2 = regression.vectorized_linearmodel(vec_X, w_vec)
    loss = regression.cost(p2, Y)
    print(w_vec, loss)
    for i in range(p2.shape[0]):
        print(p2[i], Y[i])



for counter in range(1):
    test()


