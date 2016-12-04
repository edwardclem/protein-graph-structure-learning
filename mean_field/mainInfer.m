function aucAll = mainInfer(featTest,objectsPresentTest,inferFn,thetaML)

    [N,numTest] = size(objectsPresentTest);
    nRepeat = N;
    
    nMiss = 1:N;
    
    aucAll = zeros(N,N);
    for(kk=1:numel(nMiss))
        miss = nMiss(kk);
        if(miss==N) nRepeat = 1; end; %all labels missing? just try it once

        display(['On miss: ', int2str(nMiss(kk))]);

        pDetects = cell(N,1);
        for(i=1:numTest)

            for(j=1:nRepeat)
                
                % Guarantee we mask out category j
                indMask = randperm(N);
                indMask(indMask==j) = [];
                indMask = [indMask(1:miss-1),j];
                
                gt =  objectsPresentTest(:,i);
                partialLabel = gt;
                partialLabel(indMask) = -1;

                completeLabel = inferFn(partialLabel,thetaML,featTest(:,i));
                for(jj=1:nMiss(kk))
                    mm = indMask(jj);
                    pDetects{mm}(:,end+1) = [completeLabel(mm); gt(mm)];
                end
            end
        end

        for(nn=1:N)
            aucAll(kk,nn) = areaROC(pDetects{nn}(1,:)',pDetects{nn}(2,:)');
        end
    end
end