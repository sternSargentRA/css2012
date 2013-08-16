function kf_SWR(Y, Q, R, Sm, SI, PI, T)
    # This function performs the forward kalman filter recursions for the
    # Stock-Watson-Romer model. Y is inflation
    #
    # Q,R are the SW state innovation variances
    # Sm is the standard deviation of the measurement error
    # SI,PI are the initial values for the recursions, S(1|0) and P(1|0)
    # T is the sample size

    S0 = zeros(2,T)  # current estimate of the state, S(t|t)
    S1 = zeros(2,T)  # one-step ahead estimate of the state, S(t+1|t)
    P0 = zeros(2,2,T)  # current estimate of the covariance matrix, P(t|t)
    P1 = zeros(2,2,T)  # one-step ahead covariance matrix, P(t+1|t)

    # constant parameters
    A = [0 1; 0 1]
    C = [1 0]

    # date 1
    y10 = C*SI  # E(y(t|t-1)
    D = Sm[1]
    V10 = C*PI*C' + D*D'  # V(y(t|t-1)
    S0[:,1] = SI + PI*C'*inv(V10)*( Y[:,1] - y10 )  # E(S(t|t))
    P0[:,:,1] = PI - PI*C'*inv(V10)*C*PI  # V(S(t|t))
    S1[:,1] = A*S0[:, 1]  # E(S(t+1|t)
    B = [R[2]^.5 Q[2]^.5; 0 Q[2]^.5]
    P1[:,:,1] = A*P0[:,:,1]*A' + B*B'  # V(S(t+1|t))

    for i = 2:T
        y10 = C*S1[:,i-1] # E(y(t|t-1)
        D = Sm[i]
        V10 = C*P1[:,:,i-1]*C' + D*D' # V(y(t|t-1)
        S0[:,i] = S1[:,i-1] + P1[:,:,i-1]*C'*inv(V10)*( Y[:,i] - y10 ) # E(S(t|t))
        P0[:,:,i] = P1[:,:,i-1] - P1[:,:,i-1]*C'*inv(V10)*C*P1[:,:,i-1] # V(S(t|t))
        S1[:,i] = A*S0[:,i] # E(S(t+1|t)
        B = [R[i+1]^.5 Q[i+1]^.5; 0 Q[i+1]^.5]
        P1[:,:,i] = A*P0[:,:,i]*A' + B*B' # V(S(t+1|t)
    end


    return S0,P0,P1
end


function gibbs1_swr(S0, P0, P1, T)

    # TODO: This function is *not* correct!!

    # S0,P0,P1 are outputs of the forward Kalman filter
    A = [0 1; 0 1]

    # initialize arrays for Gibbs sampler
    SA = zeros(2, T)  # artificial states
    SM = zeros(2, 1)  # backward update for conditional mean of state vector
    PM = zeros(2, 2)  # backward update for projection matrix
    P = zeros(2, 2)  # backward update for conditional variance matrix
    wa = randn(2, T)  # draws for state innovations

    # Backward recursions and sampling
    # Terminal state
    SA[:, T] = S0[:, T] + real(sqrtm(P0[:, :, T]))*wa[:, T]

    # iterating back through the rest of the sample
    for i = 1:T-1
       PM = P0[:,:,T-i]*A'*inv(P1[:,:,T-i])
       P = P0[:,:,T-i] - PM*A*P0[:,:,T-i]
       SM = S0[:,T-i] + PM*(SA[:,T-i+1] - A*S0[:,T-i])
       SA[:,T-i] = SM + real(sqrtm(P))*wa[:,T-i]
    end

    return SA
end


function svmh0(hlead,alpha,delta,sv,mu0,ss0)

    # h = svmh0(hlead,alpha,delta,sv,mu0,ss0)

    # This file returns a draw from the posterior conditional density
    # for the stochastic volatility parameter at time 0.  This is conditional
    # on the first period realization, hlead, as well as the prior and
    # parameters of the svol process.

    # mu0 and ss0 are the prior mean and variance.  The formulas simplify if
    # these are given by the unconditional mean and variance implied by the
    # state, but we haven't imposed this.  (allows for alpha = 0, delta = 1)

    # Following JPR (1994), we use a MH step, but with a simpler log-normal
    # proposal density. (Their proposal is coded in jpr.m.)

    # mean and variance for log(h) (proposal density)
    ssv = sv^2
    ss = ss0*ssv/(ssv + (delta^2)*ss0)
    mu = ss*(mu0/ss0 + delta*(log(hlead) - alpha)/ssv)

    # draw from lognormal (accept = 1, since there is no observation)
    h = exp(mu + (ss^.5)*randn(1,1))

    return h, mu, ss
end


function svmh(hlead,hlag,alpha,delta,sv,yt,hlast)

    # h = svmh(hlead,hlag,alpha,delta,sv,y,hlast)

    # This file returns a draw from the posterior conditional density
    # for the stochastic volatility parameter at time t.  This is conditional
    # on adjacent realizations, hlead and hlag, as well as the data and
    # parameters of the svol process.

    # hlast is the previous draw in the chain, and is used in the acceptance
    # step.

    # R is a dummy variable that takes a value of 1 if the trial is rejected,
    # 0 if accepted.

    # Following JPR (1994), we use a MH step, but with a simpler log-normal
    # proposal density. (Their proposal is coded in jpr.m.)

    # mean and variance for log(h) (proposal density)
    mu = alpha*(1-delta) + delta*(log(hlead)+log(hlag))/(1+delta^2)
    ss = (sv^2)/(1+delta^2)

    # candidate draw from lognormal
    htrial = exp(mu + (ss^.5)*randn(1,1))

    # acceptance probability
    lp1 = -0.5*log(htrial) - (yt^2)/(2*htrial)
    lp0 = -0.5*log(hlast) - (yt^2)/(2*hlast)
    accept = min(1,exp(lp1 - lp0))

    u = rand(1)
    if u <= accept
       h = htrial
       R = 0
    else
       h = hlast
       R = 1
    end

    return h, R
end


function svmhT(hlag,alpha,delta,sv,yt,hlast)

    # h = svmhT(hlag,alpha,delta,sv,y,hlast)

    # This file returns a draw from the posterior conditional density
    # for the stochastic volatility parameter at time T.  This is conditional
    # on the lagging realization, hlag, as well as the data and parameters
    # of the svol process.

    # hlast is the previous draw in the chain, and is used in the acceptance
    # step.

    # R is a dummy variable that takes a value of 1 if the trial is rejected,
    # 0 if accepted.

    # Following JPR (1994), we use a MH step, but with a simpler log-normal
    # proposal density. (Their proposal is coded in jpr.m.)

    # mean and variance for log(h) (proposal density)
    mu = alpha + delta*log(hlag)
    ss = sv^2

    # candidate draw from lognormal
    htrial = exp(mu + (ss^.5)*randn(1,1))

    # acceptance probability
    lp1 = -0.5*log(htrial) - (yt^2)/(2*htrial)
    lp0 = -0.5*log(hlast) - (yt^2)/(2*hlast)
    accept = min(1,exp(lp1 - lp0))

    u = rand(1)
    if u <= accept
       h = htrial
       R = 0
    else
       h = hlast
       R = 1
    end

    return h, R
end


function ig2(v0,d0,x)

    # [v,v1,d1] = ig2s(v0,d0,x)

    # This file returns posterior draw, v, from an inverse gamma with prior
    # degrees of freedom v0/2 and scale parameter d0/2.  The posterior values
    # are v1 and d1, respectively. x is a vector of innovations.

    # The simulation method follows bauwens, et al p 317.  IG2(s,v)
    #      simulate x = chisquare(v)
    #      deliver s/x

    T = size(x,1)
    v1 = v0 + T
    d1 = d0 + x'*x
    z = randn(v1,1)
    x = z'*z
    v = d1/x

    return v, v1, d1
end
