% Visualize RLLib
% @author Sam Abeyruwan

clear all;
close all;

V = load('valueFunction.txt');
E = load('continuousGridworld.txt');
P = load('continuousGridworldPath.txt');
if 0,
figure;
imagesc(fliplr(V)');
title('Off-PAC Critic Value Function: ContinuousGridworld');
colorbar;

%figure;
%mesh(V)

figure;
%subplot(1, 2, 1);
imagesc(fliplr(E)');
title('ContinuousGridworld');
axis square
colorbar;

figure;
%plot(P(:,1)*10, P(:, 2)*10, '-r', 'LineWidth', 1);
%subplot(1, 2, 2);
m = P(:, 1);
index = find(m == 2.0); % this is not good thing to do
hold on;
for i = 1 : size(index, 1) - 1
   plot(P((index(i) : index(i+1)-1), 1)*10, P((index(i) : index(i+1)-1), 2)*10, '-r', 'LineWidth', 1); 
end
title({'Off-PAC ContinuousGridworld Learned Paths', '(Avg: 37.15, (+- 95%): 0.16)'});
axis square
hold off
end
% Pendulum
S = load('swingPendulum.txt');

if 0,
numFrames = 300;
figure(1);
M = moviein(numFrames);
set(gca,'NextPlot','replacechildren');
pause;
pause(3);
for i = 1 : numFrames
   h = plot([0,sind(S(i, 1))], [0, cosd(S(i,1))], '-b', ...
                 sind(S(i, 1)), cosd(S(i,1)), '-mo',...
                'LineWidth',2,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor',[.49 1 .63],...
                'MarkerSize',10);
   axis([-2, 2, -1.5, 1.5]);
   title(['SwingPendulum step=',num2str(i)]);
   pause(0.1);
   M(:, i) = getframe;
   delete(h)
end
end

%mpgwrite(M, jet, 'movie.mpg');

% reward
figure;
plot(S([1:400], 2), '-'); 
axis([0, 400, -1, 1]);
title('SwingPendulum rewards');

