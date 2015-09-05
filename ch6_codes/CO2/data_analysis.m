%data is 15*47 records 
%the first col is the year, the second is Jan, ...., the 14th is the annual
%average and the last is the annual average with fitting values (to replace
%with the missing values)

%data2: the first col and the last 2 cols are removed

%data3: the data2 is reshaped into a col vector

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

%%