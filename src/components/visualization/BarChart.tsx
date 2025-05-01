import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface BarChartProps {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    color: string;
  }>;
  xAxisLabel?: string;
  yAxisLabel?: string;
  logarithmic?: boolean;
}

const sanitizeClassName = (label: string) => {
  return label.replace(/[\s()]/g, '-');
};

export const BarChart: React.FC<BarChartProps> = ({ 
  labels,
  datasets,
  xAxisLabel,
  yAxisLabel,
  logarithmic = false
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current) return;
    if (!datasets || datasets.length === 0) return;
    
    // Clear previous chart
    d3.select(svgRef.current).selectAll("*").remove();
    
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 20, right: 80, bottom: 40, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Find max Y value
    const maxY = d3.max(datasets, dataset => d3.max(dataset.data) || 0) || 0;
    const minY = d3.min(datasets, dataset => d3.min(dataset.data) || 0) || 0;
    
    // Create scales
    const xScale = d3.scaleBand()
      .domain(labels)
      .range([0, innerWidth])
      .padding(0.2);
    
    const groupScale = d3.scaleBand()
      .domain(datasets.map(d => d.label))
      .range([0, xScale.bandwidth()])
      .padding(0.05);
    
    const yScale = logarithmic
      ? d3.scaleLog()
          .domain([Math.max(0.1, minY), maxY * 1.1])
          .range([innerHeight, 0])
          .nice()
      : d3.scaleLinear()
          .domain([0, maxY * 1.1])
          .range([innerHeight, 0])
          .nice();
    
    // Create SVG and group for the chart
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);
    
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Add axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = logarithmic
      ? d3.axisLeft(yScale)
          .ticks(5)
          .tickFormat(d => {
            const value = Number(d);
            if (value >= 1) return value.toFixed(0);
            return value.toFixed(1);
          })
      : d3.axisLeft(yScale)
          .ticks(5);
    
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
    
    // Draw bars for each dataset
    datasets.forEach(dataset => {
      const className = `bar-${sanitizeClassName(dataset.label)}`;
      g.selectAll(`.${className}`)
        .data(dataset.data)
        .enter()
        .append("rect")
        .attr("class", className)
        .attr("x", (_, i) => (xScale(labels[i]) || 0) + groupScale(dataset.label) || 0)
        .attr("y", d => yScale(Math.max(logarithmic ? 0.1 : 0, d)))
        .attr("width", groupScale.bandwidth())
        .attr("height", d => innerHeight - yScale(Math.max(logarithmic ? 0.1 : 0, d)))
        .attr("fill", dataset.color)
        .attr("rx", 2)
        .attr("ry", 2)
        .append("title")
        .text((d, i) => `${dataset.label} (${labels[i]}): ${d}`);
    });
    
    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - margin.right + 20}, ${margin.top})`);
    
    datasets.forEach((dataset, i) => {
      const g = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`);
      
      g.append("rect")
        .attr("width", 10)
        .attr("height", 10)
        .attr("y", 4)
        .attr("rx", 2)
        .attr("ry", 2)
        .attr("fill", dataset.color);
      
      g.append("text")
        .attr("x", 20)
        .attr("y", 9)
        .attr("dy", "0.35em")
        .attr("font-size", "12px")
        .text(dataset.label);
    });
    
  }, [labels, datasets, xAxisLabel, yAxisLabel, logarithmic]);
  
  return (
    <svg ref={svgRef} className="w-full h-full"></svg>
  );
};