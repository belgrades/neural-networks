from Functions_homework1_question1_25 import generate_train_test, generate_mlp_variables, generate_rbf_variables
from Functions_homework1_question1_25 import generate_mlp_y_hat_function, generate_generic_loss_functions, generate_rbf_y_hat_function
from Functions_homework1_question1_25 import generate_generic_optimizer
from Functions_homework1_question1_25 import init_tf_session, train_generic_model
from Functions_homework1_question1_25 import predict_mlp, generate_mse_functions, compute_mse, compute_loss
from Functions_homework1_question1_25 import plot_pred_real, plot_3d_data
import numpy as np


def exercise_1_1(output_file):
    output_file.write("This is homework 1: question 1: point 1\n")
    # 1. Parameters using values of N, rho and sigma

    INPUT_UNITS = 2
    HIDDEN_UNITS = 40
    OUTPUT_UNITS = 1
    LEARNING_RATE = 0.01
    SAMPLE_SIZE = 100
    TRAIN_TEST_PROPORTION = 0.7
    RHO_CONSTANT = 0.001
    SIGMA_CONSTANT = 1.0
    GENERATE_IMAGES = True
    EPOCHS = 10000

    # 2. Generating dataset

    train_X, test_X, train_y, test_y = generate_train_test(n=SAMPLE_SIZE, p=TRAIN_TEST_PROPORTION)
    test_y = np.reshape(test_y, newshape=(int(SAMPLE_SIZE * (1 - TRAIN_TEST_PROPORTION)), OUTPUT_UNITS))
    train_y = np.reshape(train_y, newshape=(int(SAMPLE_SIZE * TRAIN_TEST_PROPORTION), OUTPUT_UNITS))

    # 3. Model definition
    X, y, omega, pi, con = generate_mlp_variables(INPUT_UNITS=INPUT_UNITS,
                                                  OUTPUT_UNITS=OUTPUT_UNITS,
                                                  HIDDEN_UNITS=HIDDEN_UNITS,
                                                  RHO_CONSTANT=RHO_CONSTANT,
                                                  SIGMA_CONSTANT=SIGMA_CONSTANT)

    y_hat = generate_mlp_y_hat_function(X, omega, pi)
    train_loss, test_loss = generate_generic_loss_functions(y_hat, y, pi, con)
    optimizer = generate_generic_optimizer(train_loss)
    sess = init_tf_session()

    # 4. Training MLP
    # TODO: add other outputs
    w_opt, b_opt, v_opt, epochs, toc = train_generic_model(sess, optimizer, train_loss, omega, X, y, train_X, train_y,
                                                 EPOCHS=EPOCHS)
    training_loss_value = compute_loss(sess, train_loss, X, y, train_X, train_y)
    print("Training objective function,%0.2f"%(training_loss_value[0]))
    output_file.write("Training objective function,%0.2f\n"%(training_loss_value[0]))

    # 5. Predictions over test set
    pred_test_y = predict_mlp(sess, y_hat, X, test_X)
    y_pred = np.array(pred_test_y).reshape(test_y.shape)
    pred_train_y = predict_mlp(sess, y_hat, X, train_X)
    #print("Predictions are: ", pred_test_y)

    # 5.1 Plot pred against
    if GENERATE_IMAGES:
        y_pred_plot = np.array(pred_test_y).reshape(test_X.shape[0])
        plot_pred_real(test_y, y_pred_plot,
                       xlab="real values of y",
                       ylab="predictions of y",
                       title="MLP full scatter plot of real values against predicted values")
        plot_3d_data(test_X, test_y, title="Real Franke2d(x1, x2) function in test data")
        plot_3d_data(test_X, y_pred_plot, title="Approximation of Franke2d(x1, x2) using MLP")

    # 5.2

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
    output_file.write("Gradient evaluations,{}\n".format(epochs))

    sess.close()
    # 7. Print to output file


def excercise_1_2(output_file):
    output_file.write("This is homework 1: question 1: point 2\n")
    # 1. Parameters using values of N, rho and sigma

    INPUT_UNITS = 2
    HIDDEN_UNITS = 50
    OUTPUT_UNITS = 1
    LEARNING_RATE = 0.01
    SAMPLE_SIZE = 100
    TRAIN_TEST_PROPORTION = 0.7
    RHO_CONSTANT = 0.00001
    SIGMA_CONSTANT = 0.01
    GENERATE_IMAGES = True
    EPOCHS = 4000

    # 2. Generating dataset

    train_X, test_X, train_y, test_y = generate_train_test(n=SAMPLE_SIZE, p=TRAIN_TEST_PROPORTION)
    test_y = np.reshape(test_y, newshape=(int(SAMPLE_SIZE * (1 - TRAIN_TEST_PROPORTION)), OUTPUT_UNITS))
    train_y = np.reshape(train_y, newshape=(int(SAMPLE_SIZE * TRAIN_TEST_PROPORTION), OUTPUT_UNITS))

    # 3. Model definition
    X, y, omega, pi, con = generate_rbf_variables(train_X=train_X,
                                                               INPUT_UNITS=INPUT_UNITS,
                                                               OUTPUT_UNITS=OUTPUT_UNITS,
                                                               HIDDEN_UNITS=HIDDEN_UNITS,
                                                               RHO_CONSTANT=RHO_CONSTANT,
                                                               SIGMA_CONSTANT=SIGMA_CONSTANT)

    y_hat = generate_rbf_y_hat_function(X, omega, pi)
    train_loss, test_loss = generate_generic_loss_functions(y_hat, y, pi, con)
    optimizer = generate_generic_optimizer(train_loss)
    sess = init_tf_session()

    # 4. Training MLP
    # TODO: add other outputs
    w_opt, c_opt, v_opt, epochs, toc = train_generic_model(sess, optimizer, train_loss, omega, X, y, train_X, train_y,
                                                           EPOCHS=EPOCHS, target="RBF")
    training_loss_value = compute_loss(sess, train_loss, X, y, train_X, train_y)
    print("Training objective function,%0.2f" % (training_loss_value[0]))
    output_file.write("Training objective function,%0.2f\n" % (training_loss_value[0]))

    # 5. Predictions over test set
    pred_test_y = predict_mlp(sess, y_hat, X, test_X)
    y_pred = np.array(pred_test_y).reshape(test_y.shape)
    pred_train_y = predict_mlp(sess, y_hat, X, train_X)
    # print("Predictions are: ", pred_test_y)

    # 5.1 Plot pred against
    if GENERATE_IMAGES:
        y_pred_plot = np.array(pred_test_y).reshape(test_X.shape[0])
        plot_pred_real(test_y, y_pred_plot,
                       xlab="real values of y",
                       ylab="predictions of y",
                       title="RBF full scatter plot of real values against predicted values")
        plot_3d_data(test_X, test_y, title="Real Franke2d(x1, x2) function in test data")
        plot_3d_data(test_X, y_pred_plot, title="Approximation of Franke2d(x1, x2) using RBF")

    # 6. Compute MSE
    mse_training, mse_test = generate_mse_functions(y_hat, y, SAMPLE_SIZE, TRAIN_TEST_PROPORTION)
    mse_test_value = compute_mse(sess, mse_test, X, y, test_X, test_y)
    mse_training_value = compute_mse(sess, mse_training, X, y, train_X, train_y)
    print("Test MSE,%0.3f" % (mse_test_value[0]))
    output_file.write("Test MSE,%0.3f\n" % (mse_test_value[0]))
    print("Training Computing Time,%0.3f" % (toc))
    output_file.write("Training Computing Time,%0.3f\n" % (toc))
    print("Function evaluations,{}".format(epochs))
    output_file.write("Function evaluations,{}\n".format(epochs))
    print("Gradient evaluations,{}".format(epochs))
    output_file.write("Gradient evaluations,{}\n".format(epochs))

    sess.close()


if __name__ == "__main__":
    file = open("output_homework1_25.txt", "w")
    exercise_1_1(file)
    excercise_1_2(file)
