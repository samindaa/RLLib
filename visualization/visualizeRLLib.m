% Visualize RLLib
% @author Sam Abeyruwan

clear all;
close all;
clc;

V = load('valueFunction.txt');
E = load('continuousGridworld.txt');
P = load('continuousGridworldPath.txt');
if 1,
figure;
imagesc(fliplr(V)');
title('Off-PAC Critic Value Function: MCar2D');
colorbar;

%figure;
%mesh(V)
end

if 0,

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
numFrames = size(S, 1);
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
   title({['SwingPendulum step=',num2str(i)],'Behavior Policy (b)', 'Actions Selected from a Uniform Distribuation'});
   pause(0.01);
   M(:, i) = getframe;
   delete(h)
end
end

%mpgwrite(M, jet, 'movie.mpg');

% reward
if 0,
figure;
plot(S((1:400), 2), '-'); 
axis([0, 400, -1, 1]);
title('SwingPendulum rewards');
end

if 0,
   u = load('mcar.txt');
   x = (-1.2:0.1:0.6);
   figure;
   pause;
   pause(3);
   for i = 1 : size(u, 1)
      h = plot(x, sin(3*x), '-b', ...
               u(i), sin(3*u(i)), '-mo',...
                'LineWidth',2,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor',[.49 1 .63],...
                'MarkerSize',10); 
     axis([-1.2, 0.6, -1, 1]);
     %title({'An Optimal Policy MCar2D', '(Avg: 111.7, (+- 95%): 0.12)'});
     title({'MCar2D','Behavior Policy (b)', 'Actions Selected from a Uniform Distribuation'});
     pause(0.005);
     delete(h);
   end
   
end

if 0,
    h = figure(1);
   M = load('mcar3D.txt');
   x = (-1.2:0.05:0.6);
   [X, Y] = meshgrid(x, x);
   Z = sin(3 * X) + sin(3 * Y);   
   view([30 50]);
   surf(X, Y, Z);
   pause;
   pause(10);
   hold on;   
   for k = 1 : size(M, 1)
       h2 = plot3(M(k, 1), M(k,2),(sin(3*M(k, 1))+sin(3*M(k, 2))), ...
                 '-mo',...
                 'LineWidth',2,...
                 'MarkerEdgeColor','k',...
                 'MarkerFaceColor',[.49 1 .63],...
                 'MarkerSize',10);    
             axis([-1.2, 0.6, -1.2, 0.6, -2, 2]);
             title('MCar3D: An Optimal Policy');
             pause(0.005);
             delete(h2);
   end
   hold off;
%    for x1 = -1 : 0.1 : 1
%        for x2 = -1: 0.1: 1           
%            h2= plot3(x1,x2,(sin(3*x1)+sin(3*x2)), '-mo',...
%                 'LineWidth',2,...
%                 'MarkerEdgeColor','k',...
%                 'MarkerFaceColor',[.49 1 .63],...
%                 'MarkerSize',10);            
%             pause(0.1);
%             delete(h2);
%        end
%    end
end
