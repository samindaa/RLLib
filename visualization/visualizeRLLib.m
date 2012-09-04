% Visualize RLLib
% @author Sam Abeyruwan

clear all;
close all;

V = load('valueFunction.txt');
E = load('env.txt');
figure;
imagesc(fliplr(V)');
colorbar;

%figure;
%mesh(V)

figure;
imagesc(fliplr(E)');
colorbar;