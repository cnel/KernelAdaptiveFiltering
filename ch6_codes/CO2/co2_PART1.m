%Copyright
%Weifeng Liu CNEL
%July 4 2008
%
%Description:
%Data analysis of CO2 data
%
%Usage:
%Ch5, CO2 concentration forecasting, figure 5-10
%
%ouside functions called
%NONE

clear all, close all
clc

load CO2_data.mat date ave_missing ave_interpolate

figure;
plot(date,ave_interpolate,'LineWidth', 2)

grid on
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

xlabel('year'),ylabel('CO2 concentration (ppmv)')
axis tight
