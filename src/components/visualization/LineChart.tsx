import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface LineChartProps {
  data?: number[];
  labels?: string[] | number[];
  color?: string;
  yAxisMin?: number;
  yAxisMax?: number;
  formatYAxis?: (value: number) => string;
  datasets?: Array<{
    data: number[];
    label: string;
    color: string;
  }>;
  xAxisLabel?: string;
  yAxisLabel?: string;
}

export const LineChart: React.FC<LineChartProps> = ({ 
  data,
  labels,
  color = "#3B82F6",
  yAxisMin,
  yAxisMax,
  formatYAxis,
  datasets,
  xAxisLabel,
  yAxisLabel
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current) return;
    if ((!data || data.length === 0) && (!datasets || datasets.length === 0)) return;
    
    // Prepare data
    const chartData = datasets || (data && labels ? [{ data, label: 'Data', color }] : []);
    if (chartData.length === 0) return;
    
    // Use the first dataset's labels if no labels provided
    const xLabels = labels || [...Array(chartData[0].data.length).keys()].map(i => i.toString());
    
    // Clear previous chart
    d3.select(svgRef.current).selectAll("*").remove();
    
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 20, right: 80, bottom: 40, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Find min/max Y values
    let minY = yAxisMin !== undefined ? yAxisMin : 
      d3.min(chartData, dataset => d3.min(dataset.data) || 0) || 0;
    let maxY = yAxisMax !== undefined ? yAxisMax : 
      d3.max(chartData, dataset => d3.max(dataset.data) || 0) || 0;
    
    // Add some padding to the y domain
    const yPadding = (maxY - minY) * 0.1;
    if (yAxisMin === undefined) minY = Math.max(0, minY - yPadding);
    if (yAxisMax === undefined) maxY = maxY + yPadding;
    
    // Create scales
    const xScale = d3.scaleBand()
      .domain(xLabels.map(d => d.toString()))
      .range([0, innerWidth])
      .padding(0.1);
    
    const yScale = d3.scaleLinear()
      .domain([minY, maxY])
      .range([innerHeight, 0]);
    
    // Create SVG and group for the chart
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);
    
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Add axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => formatYAxis ? formatYAxis(Number(d)) : d.toString());
    
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll("text")
      .attr("font-size", "10px");
    
    g.append("g")
      .call(yAxis)
      .selectAll("text")
      .attr("font-size", "10px");
    
    // Add axis labels if provided
    if (xAxisLabel) {
      g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight + 35)
        .attr("text-anchor", "middle")
        .attr("font-size", "12px")
        .text(xAxisLabel);
    }
    
    if (yAxisLabel) {
      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerHeight / 2)
        .attr("y", -40)
        .attr("text-anchor", "middle")
        .attr("font-size", "12px")
        .text(yAxisLabel);
    }
    
    // Add grid lines
    g.append("g")
      .attr("class", "grid")
      .attr("opacity", 0.1)
      .call(d3.axisLeft(yScale)
        .ticks(5)
        .tickSize(-innerWidth)
        .tickFormat(() => "")
      );
    
    // Create line generator
    const lineGenerator = d3.line<number>()
      .x((_, i) => xScale(xLabels[i].toString()) || 0 + xScale.bandwidth() / 2)
      .y(d => yScale(d));
    
    // Draw each dataset line
    chartData.forEach((dataset, index) => {
      // Draw line
      g.append("path")
        .datum(dataset.data)
        .attr("fill", "none")
        .attr("stroke", dataset.color)
        .attr("stroke-width", 2)
        .attr("d", lineGenerator);
      
      // Add dots
      g.selectAll(`.dot-${index}`)
        .data(dataset.data)
        .enter()
        .append("circle")
        .attr("cx", (_, i) => (xScale(xLabels[i].toString()) || 0) + xScale.bandwidth() / 2)
        .attr("cy", d => yScale(d))
        .attr("r", 4)
        .attr("fill", dataset.color)
        .attr("stroke", "#fff")
        .attr("stroke-width", 1)
        .append("title")
        .text((d, i) => `${dataset.label}: ${d} (${xLabels[i]})`);
    });
    
    // Add legend if multiple datasets
    if (chartData.length > 1) {
      const legend = svg.append("g")
        .attr("transform", `translate(${width - margin.right + 20}, ${margin.top})`);
      
      chartData.forEach((dataset, i) => {
        const g = legend.append("g")
          .attr("transform", `translate(0, ${i * 20})`);
        
        g.append("line")
          .attr("x1", 0)
          .attr("y1", 9)
          .attr("x2", 20)
          .attr("y2", 9)
          .attr("stroke", dataset.color)
          .attr("stroke-width", 2);
        
        g.append("circle")
          .attr("cx", 10)
          .attr("cy", 9)
          .attr("r", 3)
          .attr("fill", dataset.color);
        
        g.append("text")
          .attr("x", 25)
          .attr("y", 9)
          .attr("dy", "0.35em")
          .attr("font-size", "12px")
          .text(dataset.label);
      });
    }
    
  }, [data, labels, color, yAxisMin, yAxisMax, formatYAxis, datasets, xAxisLabel, yAxisLabel]);
  
  return (
    <svg ref={svgRef} className="w-full h-full"></svg>
  );
};