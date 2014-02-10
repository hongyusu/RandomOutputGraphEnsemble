
%%
function run_MAV_AMMlearner(filename,graph_type,t)
    
    isTest = 0;

    %% add path of libsvm
    addpath '~/softwares/libsvm-3.12/matlab/'
    addpath '../shared_scripts/'

    [~,comres]=system('hostname');
    if strcmp(comres(1:4),'dave') | strcmp(comres(1:4),'ukko') | strcmp(comres(1:4),'node')
        X=dlmread(sprintf('/home/group/urenzyme/workspace/data/%s_features',filename));
        Y=dlmread(sprintf('/home/group/urenzyme/workspace/data/%s_targets',filename));
    else
        X=dlmread(sprintf('../shared_scripts/test_data/%s_features',filename));
        Y=dlmread(sprintf('../shared_scripts/test_data/%s_targets',filename));
    end

    rand('twister', 0);


    %% parpare
    % example selection with meaningful features
    Xsum=sum(X,2);
    X=X(Xsum~=0,:);
    Y=Y(Xsum~=0,:);
    MAVsuffix=sprintf('%s_%s_%s_MAVlearner', filename,graph_type,t);
    AMMsuffix=sprintf('%s_%s_%s_AMMlearner', filename,graph_type,t);
    t=eval(t);
    % label selection with two labels
    Yuniq=[];
    for i=1:size(Y,2)
        if size(unique(Y(:,i)),1)>1
            Yuniq=[Yuniq,i];
        end
    end
    Y=Y(:,Yuniq);


    %% feature normalization (tf-idf for text data, scale and centralization for other numerical features)
    if or(strcmp(filename,'medical'),strcmp(filename,'enron')) 
        X=tfidf(X);
    elseif ~(strcmp(filename(1:2),'to'))
        X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    end

    %% change Y from -1 to 0: labeling (0/1)
    Y(Y==-1)=0;

    %%
    if isTest==1
        X=X(1:100,:);
        Y=Y(1:100,:);
    end

    % length of x and y
    Nx = length(X(:,1));
    Ny = length(Y(1,:));
    % stratified cross validation index
    nfold = 5;
    % n-fold index
    Ind = getCVIndex(Y,nfold);
    % performance
    perf=[];


    %% get dot product kernels from normalized features or just read precomputed kernels
    if or(strcmp(filename,'fp'),strcmp(filename,'cancer'))
        K=dlmread(sprintf('/home/group/urenzyme/workspace/data/%s_kernel',filename));
    else
        K = X * X'; % dot product
        K = K ./ sqrt(diag(K)*diag(K)');    %normalization diagonal is 1
    end


    %% use results from parameter selection -> mmcrf_(c,g,i)
    load(sprintf('../outputs/%s_%s_baselearner_parameters.mat',filename,graph_type));
    mmcrf_c=selected_parameters(1);
    mmcrf_g=selected_parameters(2);
    mmcrf_i=selected_parameters(3);
    mmcrf_ssc=selected_parameters(4);
    selRes
    mmcrf_c
    mmcrf_g
    mmcrf_i
    mmcrf_ssc
    
    %%
    Nrep=200;
    ensemble_size=180;

        
    %% pick up top base learner index from week to strong -> baselearner_index
    bl_results=zeros(Nrep,2);
    for i=1:Nrep
        try
            load(sprintf('../outputs/%s_%s_%d_baselearner.mat', filename,graph_type,i));
        catch err
            %disp(err)
            perf=[0];
        end
        bl_results(i,:) = [perf(1),i];
    end
    [u,v] = sort(bl_results(:,1),'descend');
    bl_results=bl_results(v(Nrep:-1:1),:); 
    bl_results = bl_results(bl_results(:,1)~=0,:); 
    baselearner_index=bl_results(:,2);
    
   

    
    
    %% MAV ensemble
    perf = [];
    ensemble_Ypred = Y*0;
    ensemble_YpredVal = Y*0;
    
    for j=1:t
        load(sprintf('../outputs/%s_%s_%d_baselearner.mat', filename,graph_type,baselearner_index(j)));
        ensemble_Ypred = ensemble_Ypred + Ypred;
        ensemble_YpredVal = ensemble_YpredVal + YpredVal;
    end
        [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,(ensemble_Ypred>=0),ensemble_YpredVal);
        perf = [acc,vecacc,pre,rec,f1,auc1,auc2]
        save(sprintf('../outputs/%s.mat', MAVsuffix), 'perf','ensemble_Ypred', 'ensemble_YpredVal');
        
        [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,(ensemble_YpredVal>=0),ensemble_YpredVal);
        perf = [acc,vecacc,pre,rec,f1,auc1,auc2]
        save(sprintf('../outputs/%s.mat', AMMsuffix), 'perf','ensemble_Ypred', 'ensemble_YpredVal');
    
    
      

    if strcmp(comres(1:4),'dave') | strcmp(comres(1:4),'ukko') | strcmp(comres(1:4),'node')
        exit
    end
end






