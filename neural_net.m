%Neural Network
no_hidden_neurons = input('enter number of neurons:');
no_outputs = 1;
epochs = input('enter no of epochs:');
alpha = 0.01;
%defining function Y=1/X
X = 1:100;
Y = [1./X];
X = X';
Y = Y';
no_inputs = size(X)
no_inputs = no_inputs(2)+1

theta1 = zeros(no_hidden_neurons,no_inputs);
theta2 = zeros(no_outputs,no_hidden_neurons+1);
N = length(X);
loss_history = zeros(1,epochs);
X = [ones(N,1) X];
X_train = X(1:80,1:end);
Y_train = Y(1:80);
X_test = X(81:100,1:end);
Y_test = Y(81:100,1:end);
N_train = length(X_train);
N_test = length(X_test);
%start training
for epoch=1:epochs
    z1 = X_train*theta1';
    a1 = sigmoid(z1);
    a1 = [ones(N_train,1) a1];
    z2 = a1*theta2';
    output = sigmoid(z2);

    %backprop

    e = Y_train-output;
    del3 = e;
    del2 = del3*theta2;
    del2n = del2(:,2:end);
    del2n = del2n.*sigmoidGradient(z1);
    gradTheta2 = del3'*a1;
    gradTheta1 = del2n'*X_train;
    gradTheta1 = gradTheta1./N;
    gradTheta2 = gradTheta2./N;

    %gradient upgrade
    theta1 = theta1+alpha*gradTheta1;
    theta2 = theta2+alpha*gradTheta2;
   
    %calculating loss after end of epoch
    z1 = X_train*theta1';
    a1 = sigmoid(z1);
    a1 = [ones(N_train,1) a1];
    z2 = a1*theta2';
    output = sigmoid(z2);
    e = Y_train-output;
    loss = sum(sum(e.*e))/N_train
    loss_history(epoch)=loss;
    fprintf('epoch %d of %d epochs loss: %f',epoch,epochs,loss_history(epoch));
end

%prediction on test data
z1 = X_test*theta1';
a1 = sigmoid(z1);
a1 = [ones(N_test,1) a1];
z2 = a1*theta2';
output = sigmoid(z2);

%calculating error on test dataset
error = Y_test-output;
error = sum(error);

%predicting function on whole dataset to compare with original function
z1 = X*theta1';
a1 = sigmoid(z1);
a1 = [ones(N,1) a1];
z2 = a1*theta2';
Ypred = sigmoid(z2);
Ypred = Ypred';


fprintf('\nerror of trained network on test set:%f',error);

figure;
epoch = 1:epochs;
plot(epoch,loss_history);
title('loss');
xlabel('epochs');
ylabel('loss');

X = 1:100;
Y = [1./X];

figure;
plot(X,Y);
title('original function 1/x');
xlabel('X');
ylabel('Y');

figure;
title('learned function');
plot(X,Ypred);
xlabel('X');
ylabel('Ypred of learned function');


    
    




