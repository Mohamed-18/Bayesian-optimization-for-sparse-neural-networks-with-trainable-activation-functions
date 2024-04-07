import numpy as np
import tensorflow as tf
from numpy import linalg as LA
from scipy.stats import invgamma

def sampling_fn(mymodel, x_train, y_train, ComputeProx, nsHMC_PlugPlay, Compute_K, Compute_E_theta,accracy, x_test, y_test) :
    c_initial = np.random.uniform(0, 1)
    lambda_b = 1
    exp_value = np.random.exponential(scale=1 / lambda_b)
    lambda_initial = exp_value / (exp_value + 1)

    max_val, min_val = 1, -1
    range_size = (max_val - min_val)
    gamma_initial = np.random.randn(1) * range_size + min_val

    model = mymodel(c_initial, lambda_initial, gamma_initial)

    Leap = 10
    varp = 1
    epsilon = 0.1
    acceptation = 0
    Lambda = 0.5
    Sigma2 = 1

    concat_z = []
    y = y_train  # Ground truth
    y_hat_old = y
    zOld = concat_z
    pOld = concat_z

    epochs = 1
    loss_fn = tf.keras.losses.binary_crossentropy

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            logits = model(x_train)  # Logits for this minibatch
            loss_value = loss_fn(y_train, logits)
        gradient = tape.gradient(loss_value, model.trainable_weights)


    for i in range(len(gradient)):
        z = tf.zeros(shape=gradient[i].shape)
        concat_z.append(np.array(z))

    pr0 = ComputeProx(Lambda / Sigma2, gradient, concat_z)

    print(gradient)

    alpha_beta_w = 0.01
    Accept_AF_c = 0
    Accept_AF_lambda = 0
    Accept_AF_gamma = 0
    Accept_hp_b = 0
    Accept_hp_gamma = 0
    c_old = c_initial
    lambda_old = lambda_initial
    gamma_old = gamma_initial
    xx = 1
    c_tab = []
    b_tab = []
    gamma_tab = []
    hyper_prior_b_old = invgamma.rvs(2, scale=1)
    hyper_prior_sigma_2_old = invgamma.rvs(2, scale=1)
    lambda_weights = []
    lambda_l = []
    lambda_weights.append(len(tf.keras.backend.flatten(model.weights[0])))
    lambda_weights.append(len(tf.keras.backend.flatten(model.weights[2])))
    lambda_weights.append(len(tf.keras.backend.flatten(model.weights[4])))
    Nmcmc = 100

    for ni in range(1, Nmcmc + 1):
        print("ni", ni)
        ##  c   ##
        c_star = np.random.uniform(0, 1)

        model = mymodel(c_star, lambda_old, gamma_old)
        y_hat_star = model.predict(x_train)

        C11 = np.exp(-(LA.norm(y_hat_old - y) ** 2))
        C21 = np.exp(-(LA.norm(y_hat_star - y) ** 2))

        C31 = C11 - C21

        Decision = min(1, abs(C31))
        u = np.random.uniform(0, xx)

        if (u < Decision):
            y_hat_old = y_hat_star
            c_old = c_star
            Accept_AF_c = Accept_AF_c + 1

        c_tab.append(c_old)

        ##  b   ##
        param_lambda = 1
        exp_value = np.random.exponential(scale=1 / param_lambda)
        lambda_star = exp_value / (exp_value + 1)

        model = mymodel(c_old, lambda_star, gamma_old)
        y_hat_star = model.predict(x_train)

        C12 = np.exp(-(LA.norm(y_hat_old - y) ** 2) - (lambda_old / hyper_prior_b_old))
        C22 = np.exp(-(LA.norm(y_hat_star - y) ** 2) - (lambda_star / hyper_prior_b_old))

        C32 = C12 - C22

        Decision = min(1, abs(C32))
        u = np.random.uniform(0, xx)

        if (u < Decision):
            y_hat_old = y_hat_star
            lambda_old = lambda_star
            Accept_AF_lambda = Accept_AF_lambda + 1

        b_tab.append(lambda_star)

        ##  gamma  ##
        max_val, min_val = 0.5, -0.5
        range_size = (max_val - min_val)
        gamma_star = np.random.randn(1) * range_size + min_val

        model = mymodel(c_old, lambda_old, gamma_star)

        y_hat_star = model.predict(x_train)

        C13 = np.exp(-(LA.norm(y_hat_old - y) ** 2) - (gamma_old ** 2 / hyper_prior_sigma_2_old))
        C23 = np.exp(-(LA.norm(y_hat_star - y) ** 2) - (gamma_star ** 2 / hyper_prior_sigma_2_old))

        C33 = C13 - C23

        Decision = min(1, abs(C33))
        u = np.random.uniform(0, xx)

        if (u < Decision):
            y_hat_old = y_hat_star
            gamma_old = gamma_star
            Accept_AF_gamma = Accept_AF_gamma + 1

        gamma_tab.append(gamma_old)

        ##  lambda_b  ##
        hyper_prior_b_star = invgamma.rvs(1, scale=5)

        C15 = np.exp(-(LA.norm(y_hat_old - y) ** 2) - (abs(lambda_old) / hyper_prior_b_old)) * (
                    hyper_prior_b_old ** (-1 - alpha_beta_w))
        C25 = np.exp(-(LA.norm(y_hat_star - y) ** 2) - (abs(lambda_old) / hyper_prior_b_star)) * (
                    hyper_prior_b_star ** (-1 - alpha_beta_w))

        C35 = C15 - C25
        Decision_beta = min(1, abs(C35))
        u_b = np.random.uniform(0, xx)

        if (u_b < Decision_beta):
            y_hat_old = y_hat_star
            hyper_prior_b_old = hyper_prior_b_star
            Accept_hp_b = Accept_hp_b + 1

        ##  sigma_2  ##
        hyper_prior_sigma_2_star = invgamma.rvs(1, scale=5)

        C16 = np.exp(-(LA.norm(y_hat_old - y) ** 2) - (abs(gamma_old) / hyper_prior_sigma_2_old)) * (
                    hyper_prior_sigma_2_old ** (-1 - alpha_beta_w))
        C26 = np.exp(-(LA.norm(y_hat_star - y) ** 2) - (abs(gamma_old) / hyper_prior_sigma_2_star)) * (
                    hyper_prior_sigma_2_star ** (-1 - alpha_beta_w))

        C36 = C16 - C26
        Decision_beta = min(1, abs(C36))
        u_b = np.random.uniform(0, xx)

        if (u_b < Decision_beta):
            y_hat_old = y_hat_star
            hyper_prior_gamma_old = hyper_prior_sigma_2_star
            Accept_hp_gamma = Accept_hp_gamma + 1

        ##  lambda_l  ##
        hyper_prior_4_star_1 = invgamma.rvs(lambda_weights[0], scale=5)
        lambda_l.append(hyper_prior_4_star_1)
        lambda_l.append(1)
        hyper_prior_4_star_2 = invgamma.rvs(lambda_weights[1], scale=5)
        lambda_l.append(hyper_prior_4_star_2)
        lambda_l.append(1)
        hyper_prior_4_star_2 = invgamma.rvs(lambda_weights[2], scale=5)
        lambda_l.append(hyper_prior_4_star_2)
        lambda_l.append(1)
        lambda_l.append(1)
        lambda_l.append(1)
        lambda_l.append(1)
        lambda_l.append(1)

        ##  W  ##
        zStar, pStar = nsHMC_PlugPlay(nb_param,zOld, varp, epsilon, pr0, concat_z, Leap)

        model.set_weights(np.array(zStar))
        y_hat_star = model(x_train)

        #y_hat_star = Compute_y_hat(zStar, x_train)

        K0 = Compute_K(pOld)
        KStar = Compute_K(pStar)
        LikelihoodRatio = Compute_E_theta(y_hat_star, y, zStar, Lambda / Sigma2, lambda_l) - Compute_E_theta(y_hat_old,
                                                                                                             y, zOld,
                                                                                                             Lambda / Sigma2,
                                                                                                             lambda_l)

        mae = tf.keras.losses.MeanSquaredLogarithmicError()
        accuracy_train = accracy(y_hat_star, y_train)
        loss_train = mae(y_train, model.predict(x_train)).numpy()

        model.set_weights(zStar)
        predtest = accracy(model.predict(x_test), y_test)
        mae = tf.keras.losses.MeanSquaredLogarithmicError()
        losstest = mae(y_test, model.predict(x_test)).numpy()

        alpha = min(1, np.exp(K0 - KStar + LikelihoodRatio))

        u = np.random.uniform(0, 0.01)

        if u < alpha:
            acceptation += 1
            print("acceptation", acceptation)
            zOld = zStar
            pOld = pStar
            y_hat_old = y_hat_star
            loss_old = loss_train
            acc_old = np.array(accuracy_train)
            pred_test_old = predtest
            loss_test_old = losstest

            print("accept w")
            print("loss train", loss_old)
            print("accuracy train", acc_old)
            print("loss test", loss_test_old)
            print("accuracy test", pred_test_old)
            print("**********")
            
            
