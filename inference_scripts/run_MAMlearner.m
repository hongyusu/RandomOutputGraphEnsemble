
%%
function run_MAMlearner(filename,graph_type,t)

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
    suffix=sprintf('%s_%s_%s_MAMlearner', filename,graph_type,t);
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
    mmcrf_c
    mmcrf_g
    mmcrf_i
    mmcrf_ssc
    
    %%
    Nrep=200;

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
    
   

    %% generate tree with random number, select according to baselearner_index -> Elist
    rand('twister', 0);
    % generate random graph
    Nnode=size(Y,2);
    Elist=cell(Nrep,1);
    for i=1:Nrep
        if strcmp(graph_type,'tree')
            E=randTreeGenerator(Nnode); % generate
        end
        if strcmp(graph_type,'pair')
            E=randPairGenerator(Nnode); % generate
        end
        E=[E,min(E')',max(E')'];E=E(:,3:4); % arrange head and tail
        E=sortrows(E,[1,2]); % sort by head and tail
        Elist{i}=E; % put into cell array
    end
    Elist=Elist(baselearner_index);
    % get new E
    Enew = [];
    for j=1:t
        Enew=[Enew;Elist{j}];
    end
    Enew=unique(Enew,'rows');
    E=Enew;
    clear Enew;

    
    %% restore mu from base learner
    muListNew=cell(nfold,1);
    for k=1:nfold
        muListNew{k}=zeros(4*size(E,1),sum(Ind~=k));
    end
    for j=1:t
        load(sprintf('../outputs/%s_%s_%d_baselearner.mat', filename,graph_type,baselearner_index(j)));
        for k=1:nfold
            muListNew{k} = muListNew{k} + mu_complete_zero(muList{k},Elist{j},E,mmcrf_c);
        end
    end    
    for k=1:nfold
        muListNew{k}=muListNew{k}/t;
    end
    
    clear muList;
    
 
    %% running
    Ypred=Y;
    YpredVal=Y;
    running_times=zeros(nfold,1);
    ensemble_muList=cell(nfold,1);

    % nfold cross validation
    for k=1:nfold
        paramsIn.mlloss         = 0;        % assign loss to microlabels(0) edges(1)
        paramsIn.profiling      = 1;        % profile (test during learning)
        paramsIn.epsilon        = mmcrf_g;        % stopping criterion: minimum relative duality gap
        paramsIn.C              = mmcrf_c;        % margin slack
        paramsIn.max_CGD_iter   = 1;		% maximum number of conditional gradient iterations per example
        paramsIn.max_LBP_iter   = 3;        % number of Loopy belief propagation iterations
        paramsIn.tolerance      = 1E-10;    % numbers smaller than this are treated as zero
        paramsIn.profile_tm_interval = 10;  % how often to test during learning
        paramsIn.maxiter        = mmcrf_i;        % maximum number of iterations in the outer loop
        paramsIn.ssc            = mmcrf_ssc;      % step size constant
        paramsIn.verbosity      = 1;
        paramsIn.debugging      = 3;
        if isTest
            paramsIn.extra_iter     = 0;        % extra iteration through examples when optimization is over
        else
            paramsIn.extra_iter     = 1;        % extra iteration through examples when optimization is over
        end
        paramsIn.filestem       = sprintf('%s',suffix);		% file name stem used for writing output

        % nfold cross validation
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        gKx_tr = K(Itrain, Itrain);     % kernel
        gKx_ts = K(Itest,  Itrain)';
        gY_tr = Y(Itrain,:); gY_tr(gY_tr==0)=-1;    % training label
        gY_ts = Y(Itest,:); gY_ts(gY_ts==0)=-1;
        % set input data
        dataIn.E = E;               % edge
        dataIn.Kx_tr = gKx_tr;      % kernel
        dataIn.Kx_ts = gKx_ts;
        dataIn.Y_tr = gY_tr;        % label
        dataIn.Y_ts = gY_ts;
        % get mu
        muNew = muListNew{k};
        % running
        [rtn,~] = baselearner(paramsIn,dataIn,muNew);
        % save margin dual mu
        ensemble_muList{k}=rtn;
        % collecting results
        load(sprintf('/var/tmp/Ypred_%s.mat', paramsIn.filestem));
        
        Ypred(Itest,:)=Ypred_ts;
        YpredVal(Itest,:)=Ypred_ts_val;
        running_times(k,1) = running_time;
    end

    
    % auc & roc random model
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,(Ypred==1),YpredVal);
    perf = [acc,vecacc,pre,rec,f1,auc1,auc2]
    
    %% need to save: Ypred, YpredVal, running_time, mu for current baselearner t,filename
    save(sprintf('../outputs/%s.mat', paramsIn.filestem), 'perf','Ypred', 'YpredVal', 'running_times', 'ensemble_muList');

    if strcmp(comres(1:4),'dave') | strcmp(comres(1:4),'ukko') | strcmp(comres(1:4),'node')
        exit
    end
end






