from Functions_homework1_question3_25 import generate_train_test, generate_decomposition_variables
from Functions_homework1_question3_25 import generate_decomposition_y_hat_function, generate_generic_loss_functions
from Functions_homework1_question3_25 import generate_decomposition_optimizers
from Functions_homework1_question3_25 import init_tf_session, train_decomposition_model
from Functions_homework1_question3_25 import predict_decomposition, generate_mse_functions, compute_mse, compute_loss
from Functions_homework1_question3_25 import plot_pred_real, plot_3d_data
from Functions_homework1_question3_25 import generate_decomposition_variables
import numpy as np


def exercise_3_1(output_file):
    output_file.write("This is homework 1: question 3\n")
    # 1. Parameters using values of N, rho and sigma

    INPUT_UNITS = 2  # Number of input nodes: 4 features and 1 bias
    HIDDEN_UNITS = 40  # Number of hidden nodes
    OUTPUT_UNITS = 1  # Number of outcomes (3 iris flowers)
    LEARNING_RATE = 0.01
    SAMPLE_SIZE = 100
    TRAIN_TEST_PROPORTION = 0.7
    RHO_CONSTANT = 0.0001
    SIGMA_CONSTANT = 1
    GENERATE_IMAGES = True
    EPOCHS = 5000

    # 2. Generating dataset

    train_X, test_X, train_y, test_y = generate_train_test(n=SAMPLE_SIZE, p=TRAIN_TEST_PROPORTION)
    test_y = np.reshape(test_y, newshape=(int(SAMPLE_SIZE * (1 - TRAIN_TEST_PROPORTION)), OUTPUT_UNITS))
    train_y = np.reshape(train_y, newshape=(int(SAMPLE_SIZE * TRAIN_TEST_PROPORTION), OUTPUT_UNITS))

    # 3. Model definition
    X, y, omega, pi, con = generate_decomposition_variables(INPUT_UNITS=INPUT_UNITS,
                                                            OUTPUT_UNITS=OUTPUT_UNITS,
                                                            HIDDEN_UNITS=HIDDEN_UNITS,
                                                            RHO_CONSTANT=RHO_CONSTANT,
                                                            SIGMA_CONSTANT=SIGMA_CONSTANT)

    y_hat = generate_decomposition_y_hat_function(X, omega, pi)
    train_loss, test_loss = generate_generic_loss_functions(y_hat, y, pi, con)
    convex, non_convex = generate_decomposition_optimizers(train_loss, omega, LEARNING_RATE=LEARNING_RATE)
    sess = init_tf_session()

    # 4. Training MLP
    # TODO: add other outputs
    w_opt, b_opt, v_opt, epochs, toc = train_decomposition_model(sess, convex, non_convex, train_loss, omega, X, y, train_X, train_y,
                                                           EPOCHS=EPOCHS)
    training_loss_value = compute_loss(sess, train_loss, X, y, train_X, train_y)
    print("Training objective function,%0.2f"%(training_loss_value[0]))
    output_file.write("Training objective function,%0.2f\n"%(training_loss_value[0]))

    # 5. Predictions over test set
    pred_test_y = predict_decomposition(sess, y_hat, X, test_X)
    y_pred = np.array(pred_test_y).reshape(test_y.shape)
    pred_train_y = predict_decomposition(sess, y_hat, X, train_X)
    #print("Predictions are: ", pred_test_y)

    # 5.1 Plot pred against
    if GENERATE_IMAGES:
        y_pred_plot = np.array(pred_test_y).reshape(test_X.shape[0])
        plot_pred_real(test_y, y_pred_plot,
                       xlab="real values of y",
                       ylab="predictions of y",
                       title="Decomposition scatter plot of real values against predicted values")
        plot_3d_data(test_X, test_y)
        plot_3d_data(test_X, y_pred_plot, title="Approximation of the franke2d function via Decomposition methods")

    # 6. Compute MSE
    mse_training, mse_test = generate_mse_functions(y_hat, y, SAMPLE_SIZE, TRAIN_TEST_PROPORTION)
    mse_test_value = compute_mse(sess, mse_test, X, y, test_X, test_y)
    mse_training_value = compute_mse(sess, mse_training, X, y, train_X, train_y)
    print("Test MSE,%0.3f"%(mse_test_value[0]))
    output_file.write("Test MSE,%0.3f\n"%(mse_test_value[0]))
    print("Training Computing Time,%0.3f"%(toc))
    output_file.write("Training Computing Time,%0.3f\n"%(toc))
    print("Function evaluations,{}".format(epochs))
    output_file.write("Function evaluations,{}\n".format(epochs))
    print("Gradient evaluations,{}".format(epochs))
    output_file.write("Gradient evaluations,{}\n".format(2*epochs))
    # 7. Print to output file


if __name__ == "__main__":
    file = open("output_homework1_25.txt", "a")
    exercise_3_1(file)