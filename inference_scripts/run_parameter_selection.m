
%%
%
% running parameter selection for dataset 'filename' with 'graph_type' as
% output graph structure
function run_parameter_selection(filename,graph_type,isTest)

    %% tackle input parameters
    if nargin <1
        disp('Not enough input parameters!')
        return;
    end
    if nargin < 2
        graph_type = 'tree';
    end
    if nargin < 3
        isTest = '1';
    end
    % set random number seed
    rand('twister', 0);
    % suffix for write result files
    suffix=sprintf('%s_%s_baselearner', filename,graph_type);
    %
    isTest = eval(isTest);
    % get search path
    addpath('../shared_scripts/');  
    % get current hostname
    [~,comres]=system('hostname');
    if strcmp(comres(1:4),'dave') | strcmp(comres(1:4),'ukko') | strcmp(comres(1:4),'node')
        X=dlmread(sprintf('/home/group/urenzyme/workspace/data/%s_features',filename));
        Y=dlmread(sprintf('/home/group/urenzyme/workspace/data/%s_targets',filename));
    else
        X=dlmread(sprintf('../shared_scripts/test_data/%s_features',filename));
        Y=dlmread(sprintf('../shared_scripts/test_data/%s_targets',filename));
    end


    %% data preprocessing
    % select example with features that make sense
    Xsum=sum(X,2);
    X=X(Xsum~=0,:);
    Y=Y(Xsum~=0,:);
    % label selection with two classes
    Yuniq=zeros(1,size(Y,2));
    for i=1:size(Y,2)
        if size(unique(Y(:,i)),1)>1
            Yuniq(i)=i;
        end
    end
    Y=Y(:,Yuniq(Yuniq~=0));


    %% feature normalization (tf-idf for text data, scale and centralization for other numerical features)
    if or(strcmp(filename,'medical'),strcmp(filename,'enron')) 
        X=tfidf(X);
    elseif ~(strcmp(filename(1:2),'to'))
        X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    end

    %% change Y from -1 to 0: labeling (0/1)
    Y(Y==-1)=0;

    % stratified cross validation index
    nfold = 5;
    % n-fold index
    Ind = getCVIndex(Y,nfold);


    %% get dot product kernels from normalized features or just read precomputed kernels
    if or(strcmp(filename,'fpuni'),strcmp(filename,'cancer'))
        if strcmp(comres(1:4),'dave') | strcmp(comres(1:4),'ukko') | strcmp(comres(1:4),'node')
            K=dlmread(sprintf('/home/group/urenzyme/workspace/data/%s_kernel',filename));
        else
            K=dlmread(sprintf('../shared_scripts/test_data/%s_kernel',filename));
        end
    else
        K = X * X'; % dot product
        K = K ./ sqrt(diag(K)*diag(K)');    %normalization diagonal is 1
    end

    %% select part of the data for code sanity check
    if isTest==1
        X=X(1:100,:);
        Y=Y(1:100,1:10);
        K=K(1:100,1:100);
    end

    %% running base learning for parameter selection
    % not 5-fold cross validation but rather do learning on 4-fold and
    % inference on the rest 1-fold
    %
    mmcrf_cs=[50,10,5,1,0.5,0.1,0.01];
    mmcrf_gs=[0.9,0.8,0.7,0.6];
    mmcrf_is=[3,5,10,15,20];
    Isel = randsample(1:size(K,2),ceil(size(K,2)*.5));
    IselTrain=Isel(1:ceil(numel(Isel)/5*4));
    IselTest=Isel((ceil(numel(Isel)/5*4+1)):numel(Isel));
    selRes=zeros(numel(mmcrf_gs)*numel(mmcrf_is),numel(mmcrf_cs));
    for l=1:numel(mmcrf_is)
    for i=1:numel(mmcrf_gs)
        for j=1:numel(mmcrf_cs)
            % set input parameters
            paramsIn.mlloss         = 0;        % assign loss to microlabels(0) edges(1)
            paramsIn.profiling      = 1;        % profile (test during learning)
            paramsIn.epsilon        = mmcrf_gs(i);        % stopping criterion: minimum relative duality gap
            paramsIn.C              = mmcrf_cs(i);        % margin slack
            paramsIn.max_CGD_iter   = 1;		% maximum number of conditional gradient iterations per example
            paramsIn.max_LBP_iter   = 3;        % number of Loopy belief propagation iterations
            paramsIn.tolerance      = 1E-10;    % numbers smaller than this are treated as zero
            paramsIn.profile_tm_interval = 10;  % how often to test during learning
            paramsIn.maxiter        = mmcrf_is(l);        % maximum number of iterations in the outer loop
            paramsIn.verbosity      = 1;
            paramsIn.debugging      = 3;
            if isTest
                paramsIn.extra_iter     = 0;        % extra iteration through examples when optimization is over
            else
                paramsIn.extra_iter     = 1;        % extra iteration through examples when optimization is over
            end
            paramsIn.filestem       = sprintf('%s',suffix);		% file name stem used for writing output
            % random seed
            rand('twister', 0);
            % generate random graph
            Nnode=size(Y,2);
            if strcmp(graph_type, 'tree')
                E=randTreeGenerator(Nnode); % generate
            end
            if strcmp(graph_type, 'pair')
                E=randPairGenerator(Nnode); % generate
            end
            E=[E,min(E')',max(E')'];E=E(:,3:4); % arrange head and tail
            E=sortrows(E,[1,2]); % sort by head and tail
            % running
            % to store results
            Ypred=Y; % one addition column for dummy node
            % nfold cross validation
            Itrain = IselTrain;
            Itest  = IselTest;
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
            % running
            [~,~] = baselearner(paramsIn,dataIn);
            % collecting results
            load(sprintf('/var/tmp/Ypred_%s.mat', paramsIn.filestem));
            Ypred(Itest,:)=Ypred_ts;
            selRes((l-1)*numel(mmcrf_gs)+i,j) = sum(sum((Ypred(IselTest,:)>=0)==Y(IselTest,:)));
            disp(selRes)
        end
    end
    end
    
    %% extract best parameters from the results and save to file
    is = find(max(selRes,[],1)==max(max(selRes,[],1)));
    mmcrf_c=mmcrf_cs(is(1));
    is = rem(find(max(selRes,[],2)==max(max(selRes,[],2))),numel(mmcrf_gs));
    is(is==0)=numel(mmcrf_gs);
    mmcrf_g=mmcrf_gs(is(1));
    is = ceil(find(max(selRes,[],2)==max(max(selRes,[],2)))/numel(mmcrf_gs)-0.01);
    mmcrf_i=mmcrf_is(is(1));

    selected_parameters=[mmcrf_c,mmcrf_g,mmcrf_i];
    disp(selected_parameters)

    save(sprintf('../outputs/%s_parameters', paramsIn.filestem),'selected_parameters','selRes');

    if ~isTest
        exit
    end
end






