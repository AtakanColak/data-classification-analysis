%hey i'm ata and i swear this is my code as not much people did it on
%matlab. however if you run it, it will only answer the last question
%because i commented plot statements for previous questions and if you want
%to check them you need to comment all the code after that plot statement

%Gib high marks pls thx

%Question 1.1
train_data = load('ac16438.train')
attribute_matrix_train = [train_data(:,3),train_data(:,4)]
class_count = 3
% plotmatrix(train_data)

%Question 1.2
[labels,centroids,sumd] = kmeans(attribute_matrix_train,class_count)
class_train_1 = get_voronoi_class(attribute_matrix_train, labels,1);
class_train_2 = get_voronoi_class(attribute_matrix_train, labels,2);
class_train_3 = get_voronoi_class(attribute_matrix_train, labels,3);
% draw_three_classes(class_train_1,class_train_2,class_train_3,'o');


% Question 1.3
test_data = load('ac16438.test')
attribute_matrix_test = [test_data(:,3),test_data(:,4)]
[result,indices] = min(transpose(pdist2(attribute_matrix_test,centroids,'euclidean'))); 
class_test_1 = get_voronoi_class(attribute_matrix_test, indices,1);
class_test_2 = get_voronoi_class(attribute_matrix_test, indices,2);
class_test_3 = get_voronoi_class(attribute_matrix_test, indices,3);
hold on
% draw_three_class_voronoi(class_train_1,class_train_2,class_train_3, class_test_1, class_test_2,class_test_3, centroids)

% % Question 1.4
% 
% [labels,centroids,sumd] = kmeans(attribute_matrix_train,3, 'MaxIter', 1)
% 
% while (sumd(1) <= (2 * (sumd(2) + sumd(3))))
%     [labels,centroids,sumd] = kmeans(attribute_matrix_train,3, 'MaxIter', 1)
% end
% 
% class_train_1 = get_voronoi_class(attribute_matrix_train, labels,1);
% class_train_2 = get_voronoi_class(attribute_matrix_train, labels,2);
% class_train_3 = get_voronoi_class(attribute_matrix_train, labels,3);
% 
% [result,indices] = min(transpose(pdist2(attribute_matrix_test,centroids,'euclidean')));
% 
% class_test_1 = get_voronoi_class(attribute_matrix_test, indices,1);
% class_test_2 = get_voronoi_class(attribute_matrix_test, indices,2);
% class_test_3 = get_voronoi_class(attribute_matrix_test, indices,3);
 draw_three_class_voronoi(class_train_1,class_train_2,class_train_3, class_test_1, class_test_2,class_test_3, centroids)



% Question 2.1

mean1 = mean(class_train_1)
mean2 = mean(class_train_2)
mean3 = mean(class_train_3)

cov1  = cov(class_train_1)
cov2  = cov(class_train_2)
cov3  = cov(class_train_3)

 
one_pdf     = my_pdf(mean1,cov1);
two_pdf     = my_pdf(mean2,cov2);
three_pdf   = my_pdf(mean3,cov3);
 

%Question 2.2

[x,y]   = meshgrid(linspace(0,10),linspace(0,10))

P1 = (1/(2*pi*sqrt(det(cov1))))*exp(-6/2);
P2 = (1/(2*pi*sqrt(det(cov2))))*exp(-6/2);
P3 = (1/(2*pi*sqrt(det(cov3))))*exp(-6/2);

D1 = two_pdf      ./ one_pdf;
D2 = three_pdf    ./ two_pdf;
D3 = one_pdf      ./ three_pdf;

hold on
contour(x,y,one_pdf,[P1,P1]);
contour(x,y,two_pdf,[P2,P2]);
contour(x,y,three_pdf,[P3,P3]);


%Question 2.3
% D1 = two_pdf      ./ (one_pdf + three_pdf);
% D2 = three_pdf    ./ (two_pdf + one_pdf);
% D3 = one_pdf      ./ (two_pdf + three_pdf) ;

[x,y]   = meshgrid(linspace(0,10),linspace(0,10))
% 
% P1 = (1/(2*pi*sqrt(det(cov1))))*exp(-6/2);
% P2 = (1/(2*pi*sqrt(det([[1,0];[0,1]]))))*exp(-6/2);
% P3 = (1/(2*pi*sqrt(det([[1,0];[0,1]]))))*exp(-6/2);

% one_pdf     = my_pdf(mean1,[[1,0];[0,1]]);
% two_pdf     = my_pdf(mean2,[[1,0];[0,1]]);
% three_pdf   = my_pdf(mean3,[[1,0];[0,1]]);


one_pdf = 2 * one_pdf;
D1 = two_pdf      ./ one_pdf ;
D2 = three_pdf    ./ two_pdf ;
D3 = one_pdf      ./ three_pdf ;

contour(x,y,D1,[1,1],'r');
contour(x,y,D2,[1,1],'g');
contour(x,y,D3,[1,1],'b');
hold off



function keyzero = my_pdf(mean, cov)
    keyzero = zeros(100,100)
    [X,Y]   = meshgrid(linspace(0,10),linspace(0,10))
    for x = 1 : 100
        z = [transpose(X(x,:)), transpose(Y(x,:))];
        keyzero(x,:) = mvnpdf(z, mean, cov);
    end
%     contour(X,Y,keyzero,[0.05,0.05])
end

function draw_three_class_voronoi(train1,train2,train3, test1, test2, test3, centroids)
    hold on
    voronoi(centroids(:,1),centroids(:,2));
    draw_three_classes(train1,train2,train3,'.');
    draw_three_classes(test1,test2,test3,'*');
    hold off
end

function draw_three_classes(class1, class2, class3, shape) 
    draw_class(class1, shape);
    draw_class(class2, shape);
    draw_class(class3, shape);
end

function draw_class(class, shape) 
    scatter(class(:,1),class(:,2),shape);
end

function class = get_voronoi_class(at_mat, clu_ind, cla_num)
   class_index = find(clu_ind == cla_num);
   class = [];
   for i = class_index
       class = [class; at_mat(i,:)];
   end
end

