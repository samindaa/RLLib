% Visualize RLLib
% @author Sam Abeyruwan

clear all;
close all;

V = load('valueFunction.txt');
E = load('continuousGridworld.txt');
P = load('continuousGridworldPath.txt');
figure;
imagesc(fliplr(V)');
colorbar;

%figure;
%mesh(V)

figure;
imagesc(fliplr(E)');
colorbar;

figure;
plot(P(:,1)*10, P(:, 2)*10, '-r', 'LineWidth', 1);

% Pendulum
S = load('swingPendulum.txt');
figure;
   
for i = 1 : size(S, 1)
   h = plot([0,sind(S(i, 1))], [0, cosd(S(i,1))], '-b');
   axis([-2, 2, -1.5, 1.5]);
   pause(0.1);
   delete(h);
end


