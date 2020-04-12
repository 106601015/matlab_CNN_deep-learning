clc;clear;
train_path='C:\Users\user\Desktop\�{���y��\DeepLearning\database\train_images\';
test_path='C:\Users\user\Desktop\�{���y��\DeepLearning\database\test_images\';
small_test_path='C:\Users\user\Desktop\�{���y��\DeepLearning\database\small_test_images\';
net = alexnet;

pre_train_ds = imageDatastore(train_path,'IncludeSubfolders',true,'LabelSource','foldernames');
pre_test_ds = imageDatastore(test_path);
pre_small_test_ds = imageDatastore(small_test_path);
train_ds = augmentedImageDatastore([227 227],pre_train_ds,'ColorPreprocessing','gray2rgb');
test_ds = augmentedImageDatastore([227 227],pre_test_ds,'ColorPreprocessing','gray2rgb');
small_test_ds = augmentedImageDatastore([227 227],pre_small_test_ds,'ColorPreprocessing','gray2rgb');
%train_ds.Labels() %�Slabels�F
numClasses = numel(categories(pre_train_ds.Labels))%numel=���ƶq

layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;

options = trainingOptions('sgdm','InitialLearnRate', 0.001,'MaxEpochs',5 , ...
    'Plots','training-progress');

[newnet,info] = trainNetwork(train_ds, layers, options);
testpreds = classify(newnet,test_ds)
testpreds2 = classify(newnet,small_test_ds)


[NUM,TXT,RAW]=xlsread('C:\Users\user\Desktop\�{���y��\DeepLearning\database\test.csv')
j=2
for i=1:3162
    buf = strsplit(test_ds.Files{i},'\')
    if buf{9} == TXT{j}
        if testpreds{j}==testpreds(6)
            TXT{10143+1+j}='0'
        elseif testpreds{j}==testpreds(2)
            TXT{10143+1+j}='1'
        elseif testpreds{j}==testpreds(1)
            TXT{10143+1+j}='2'
        elseif testpreds{j}==testpreds(10)
            TXT{10143+1+j}='3'
        elseif testpreds{j}==testpreds(12)
            TXT{10143+1+j}='4'
        elseif testpreds{j}==testpreds(4)
            TXT{10143+1+j}='5'
        end
        j=j+1;
    end
end
writecell(TXT,'C:\Users\user\Desktop\�{���y��\DeepLearning\database\test.csv')
%writematrix(TXT,'C:\Users\user\Desktop\�{���y��\DeepLearning\database\test.csv')
%���TXT�A�q1('ID')~10015�j��Astrsplit(test_ds.Files{3126},'\')��{9}���ɦW�A��Өüg�ȶi�JTXT(10143+1+i)
%�X�֥�cat(1,a,b)

%options = trainingOptions('sgdm', ...
%    'LearnRateSchedule','piecewise', ...
%    'LearnRateDropFactor',0.2, ...
%    'LearnRateDropPeriod',5, ...
%    'MaxEpochs',20, ...
%    'MiniBatchSize',64, ...
%    'Plots','training-progress')

%options = trainingOptions('sgdm', ...
%    'MaxEpochs',8, ...
%    'ValidationData',{XValidation,YValidation}, ...
%    'ValidationFrequency',30, ...
%    'Verbose',false, ...
%    'Plots','training-progress');
%net = trainNetwork(XTrain,YTrain,layers,options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%���h�d��+�}�G�@��
%ly=net.Layers();
%in_layer = ly(1)
%out_layer=ly(25)
%in_name = in_layer.Name()
%�h��k�i�d��pInputSize()
%categorynames = out_layer.Classes();
%[pred, scores] = classify(net,img1)
%bar(scores)
%highscores = scores > 0.01;
%bar(scores(highscores))
%xticklabels(categorynames(highscores))
%%%thresh = median(scores) + std(scores)
%�ʺA�H��


%�j�ƶq�Ϥ��޲z+�P�ɾާ@
%chan_path='database\test_images\test*.png';
%path=[cons_path,chan_path]
%train_ds = imageDatastore(path)
%fname = train_ds.Files()
%imgno7 = readimage(train_ds,7)
%preds = classify(net,train_ds)
%����ds�����w��
%%auds = augmentedImageDatastore([227 227],train_ds)
%auds����ӹϮw�w�B�z

%�Ƕ���m��
%montage(imds)
%���all imds�Ϥ�
%auds = augmentedImageDatastore([227 ...
%227],imds,'ColorPreprocessing','gray2rgb')
%��ӦǶ���Ʈw��rgb

%�s�x���l��Ƨ��]�j��
%flwrds = imageDatastore('Flowers','IncludeSubfolders',true)

%%�E���ǲ�=>�ݭn�w�B�z����+�w�T�{�Ϲ�+�t��k�ܶq
%flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true, ...
%    'LabelSource','foldernames')
%�N�l��Ƨ����l�]�����Ҫ��s�x
%flowernames = flwrds.Labels()
%�o�@��labels
%[flwrTrain,flwrTest] = splitEachLabel(flwrds,0.8,'randomized')
%���Φ��V�m�ո����եB�H��
%[flwrTrain,flwrTest] = splitEachLabel(flwrds,50)
%�C�Ӽ��ҵ�ds1���ƶq=50�A�קK���������v

%%%%%%%%23�h���s����X�A24�h�Ԩ̵e����%%%%%%%%
%fc = fullyConnectedLayer(12)
%�Ыإ��s���h(12��)
%layers(23)=fc
%����23�h
%layers(25)=classificationLayer
%new��25�h
%opts = trainingOptions('sgdm','InitialLearnRate',0.001)
%�ЫذV�m�ﶵ(sgdm),�B�ǲߺ�אּ0.001
%�ǲߺ�<<�h�����̨ΫD�̨ΡA�L�j�h�_��

%����
%load pathToImages
%load trainedFlowerNetwork flowernet info
%plot(info.TrainingLoss)
%�e�l���v(�]�i�HTrainingAccurary�άOBaseLearnRate)
%�̫�θշҲն]�@���ݥ��T�v

%flwrActual = testImgs.Labels
%���Tlabel
%numCorrect=nnz(flwrPreds == flwrActual)
%fracCorrect = numCorrect/numel(flwrPreds)
%��勵�T���X�ӡA�A���`�Ʈ����T�v
%confusionchart(testImgs.Labels, flwrPreds)
%�V�c�x�}����