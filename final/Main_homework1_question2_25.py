from final.Functions_homework1_question2_25 import generate_train_test, generate_extreme_mlp_variables
from final.Functions_homework1_question2_25 import generate_mlp_y_hat_function, generate_mlp_loss_functions
from final.Functions_homework1_question2_25 import generate_mlp_optimizer
from final.Functions_homework1_question2_25 import init_tensorflow_session, train_mlp
import numpy as np


if __name__ == "__main__":
    # 1. Parameters using values of N, rho and sigma

    INPUT_UNITS = 2
    HIDDEN_UNITS = 80
    OUTPUT_UNITS = 1
    LEARNING_RATE = 0.01
    SAMPLE_SIZE = 1000
    TRAIN_TEST_PROPORTION = 0.7
    RHO_CONSTANT = 0.00001
    SIGMA_CONSTANT = 4
    GENERATE_IMAGES = False
    EPOCHS = 20000

    # 2. Generating dataset

    train_X, test_X, train_y, test_y = generate_train_test(n=SAMPLE_SIZE, p=TRAIN_TEST_PROPORTION)
    test_y = np.reshape(test_y, newshape=(int(SAMPLE_SIZE * (1 - TRAIN_TEST_PROPORTION)), OUTPUT_UNITS))
    train_y = np.reshape(train_y, newshape=(int(SAMPLE_SIZE * TRAIN_TEST_PROPORTION), OUTPUT_UNITS))

    # 3. Model definition
    X, y, omega, pi, con = generate_extreme_mlp_variables(INPUT_UNITS=INPUT_UNITS,
                                                          OUTPUT_UNITS=OUTPUT_UNITS,
                                                          HIDDEN_UNITS=HIDDEN_UNITS,
                                                          RHO_CONSTANT=RHO_CONSTANT,
                                                          SIGMA_CONSTANT=SIGMA_CONSTANT)

    y_hat = generate_mlp_y_hat_function(X, omega, pi)
    train_loss, test_loss = generate_mlp_loss_functions(y_hat, y, pi, con)
    optimizer = generate_mlp_optimizer(train_loss)
    sess = init_tensorflow_session()

    # 4. Training MLP

    loss, toc = train_mlp(sess, optimizer, train_loss, X, y, train_X, train_y)
    print("Training loss: {} Time: {}".format(loss, toc))
    # 5. Predictions over test set

    # 6. Compute MSE

    # 7. Print to output file