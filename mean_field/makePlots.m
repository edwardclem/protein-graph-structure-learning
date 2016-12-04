function makePlots(algs,aucAll,llTrace,dataset,fignum)
    N=13;
    if(dataset==2) N=23; end;
    
    plotStyles = {'r.-','g.-','b.-','c.-','k.-','mx-','yx-'};
    figure(fignum);
    for(a=1:numel(algs))
        temp = mean(aucAll(:,:,a),2);
        plot(N-1:-1:0,temp,plotStyles{a}); hold on;
    end
    hold off;
    legend({algs(1:numel(algs)).name},'Location','SouthEast');
    axis([0,N,0,1]);
    xlabel('# of labels given');
    ylabel('Average AUC');
    title(sprintf('AUC vs #labels'));

    set(gcf, 'PaperPosition', [0 0 5 5]); %Position plot at left hand corner with width 5 and height 5.
    set(gcf, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
    saveas(gcf,sprintf('auc_data%d',dataset), 'pdf') %Save figure

    figure(100+fignum);
    for(a=1:numel(algs))
        plot(llTrace(:,a),plotStyles{a}); hold on;
    end
    hold off;
    xlabel('# iterations');
    ylabel('Negative training log-likelihood');
    title(sprintf('Negative Loglikelihood vs. iteration'));
    legend({algs(1:numel(algs)).name},'Location','NorthEast');

    set(gcf, 'PaperPosition', [0 0 5 5]); %Position plot at left hand corner with width 5 and height 5.
    set(gcf, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
    saveas(gcf,sprintf('loglike_data%d',dataset), 'pdf') %Save figure
end

