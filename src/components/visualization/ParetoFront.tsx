import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface ParetoFrontProps {
  instanceId: string;
  selectedSolution: number | null;
  onSelectSolution: (index: number | null) => void;
}

interface Solution {
  index: number;
  obj1: number;
  obj2: number;
  dominated: boolean;
}

const ParetoFront: React.FC<ParetoFrontProps> = ({ 
  instanceId,
  selectedSolution,
  onSelectSolution
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current) return;
    
    // Clear previous chart
    d3.select(svgRef.current).selectAll("*").remove();
    
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 50, right: 50, bottom: 60, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Generate some example data for the Pareto front
    const numSolutions = 50;
    const baseCost1 = instanceId === 'kroAB100' ? 22000 : instanceId === 'kroAB150' ? 28000 : 34000;
    const baseCost2 = instanceId === 'kroAB100' ? 24000 : instanceId === 'kroAB150' ? 30000 : 36000;
    
    // Generate non-dominated points
    const paretoPoints: Solution[] = Array.from({ length: 15 }, (_, i) => {
      const t = i / 14;  // Parameter between 0 and 1
      return {
        index: i,
        obj1: baseCost1 * (1.0 + 0.2 * t) + Math.random() * 500,
        obj2: baseCost2 * (1.2 - 0.2 * t) + Math.random() * 500,
        dominated: false
      };
    });
    
    // Generate dominated points
    const dominatedPoints: Solution[] = Array.from({ length: numSolutions - paretoPoints.length }, (_, i) => {
      const randomParetoIndex = Math.floor(Math.random() * paretoPoints.length);
      const paretoPt = paretoPoints[randomParetoIndex];
      return {
        index: i + paretoPoints.length,
        obj1: paretoPt.obj1 + Math.random() * 2000,
        obj2: paretoPt.obj2 + Math.random() * 2000,
        dominated: true
      };
    });
    
    const allPoints = [...paretoPoints, ...dominatedPoints];
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([d3.min(allPoints, d => d.obj1) * 0.99, d3.max(allPoints, d => d.obj1) * 1.01])
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([d3.min(allPoints, d => d.obj2) * 0.99, d3.max(allPoints, d => d.obj2) * 1.01])
      .range([innerHeight, 0]);
    
    // Create SVG and group for the chart
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);
    
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Add axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(5)
      .tickFormat(d => `${Math.round(Number(d) / 1000)}k`);
      
    const yAxis = d3.axisLeft(yScale)
      .ticks(5)
      .tickFormat(d => `${Math.round(Number(d) / 1000)}k`);
    
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis);
    
    g.append("g")
      .call(yAxis);
    
    // Add axis labels
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + 40)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .text("Objective 1 (Cost)");
    
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .text("Objective 2 (Cost)");
    
    // Add title
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", -20)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .text(`Pareto Front for ${instanceId}`);
    
    // Plot dominated points
    g.selectAll(".dominated-point")
      .data(dominatedPoints)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d.obj1))
      .attr("cy", d => yScale(d.obj2))
      .attr("r", 4)
      .attr("class", "pareto-point")
      .attr("opacity", 0.5)
      .on("click", (event, d) => {
        onSelectSolution(d.index);
      })
      .append("title")
      .text(d => `Solution ${d.index + 1}\nObj1: ${Math.round(d.obj1)}\nObj2: ${Math.round(d.obj2)}`);
    
    // Plot non-dominated points
    g.selectAll(".pareto-point")
      .data(paretoPoints)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d.obj1))
      .attr("cy", d => yScale(d.obj2))
      .attr("r", 6)
      .attr("class", d => d.index === selectedSolution ? "non-dominated selected" : "non-dominated")
      .attr("opacity", d => d.index === selectedSolution ? 1 : 0.8)
      .on("click", (event, d) => {
        onSelectSolution(d.index);
      })
      .append("title")
      .text(d => `Solution ${d.index + 1}\nObj1: ${Math.round(d.obj1)}\nObj2: ${Math.round(d.obj2)}`);
    
    // Connect the Pareto front points with a line
    const lineGenerator = d3.line<Solution>()
      .x(d => xScale(d.obj1))
      .y(d => yScale(d.obj2))
      .curve(d3.curveMonotoneX);
    
    const sortedParetoPoints = [...paretoPoints].sort((a, b) => a.obj1 - b.obj1);
    
    g.append("path")
      .datum(sortedParetoPoints)
      .attr("d", lineGenerator)
      .attr("fill", "none")
      .attr("stroke", "#0D9488")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "5,5")
      .attr("opacity", 0.7);
    
    // Add ideal point and nadir point
    const idealPoint = {
      obj1: d3.min(paretoPoints, d => d.obj1) || 0,
      obj2: d3.min(paretoPoints, d => d.obj2) || 0
    };
    
    const nadirPoint = {
      obj1: d3.max(paretoPoints, d => d.obj1) || 0,
      obj2: d3.max(paretoPoints, d => d.obj2) || 0
    };
    
    g.append("circle")
      .attr("cx", xScale(idealPoint.obj1))
      .attr("cy", yScale(idealPoint.obj2))
      .attr("r", 5)
      .attr("fill", "#10B981")
      .attr("stroke", "#059669")
      .attr("stroke-width", 1)
      .append("title")
      .text(`Ideal Point\nObj1: ${Math.round(idealPoint.obj1)}\nObj2: ${Math.round(idealPoint.obj2)}`);
    
    g.append("circle")
      .attr("cx", xScale(nadirPoint.obj1))
      .attr("cy", yScale(nadirPoint.obj2))
      .attr("r", 5)
      .attr("fill", "#EF4444")
      .attr("stroke", "#DC2626")
      .attr("stroke-width", 1)
      .append("title")
      .text(`Nadir Point\nObj1: ${Math.round(nadirPoint.obj1)}\nObj2: ${Math.round(nadirPoint.obj2)}`);
    
    // Add a legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - margin.right - 150}, ${margin.top})`);
    
    const legendItems = [
      { color: "#14B8A6", label: "Pareto Optimal" },
      { color: "#F97316", label: "Dominated Solution" },
      { color: "#10B981", label: "Ideal Point" },
      { color: "#EF4444", label: "Nadir Point" }
    ];
    
    legendItems.forEach((item, i) => {
      const g = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`);
      
      g.append("circle")
        .attr("r", 6)
        .attr("fill", item.color);
      
      g.append("text")
        .attr("x", 15)
        .attr("y", 4)
        .text(item.label)
        .attr("font-size", "12px");
    });
    
  }, [instanceId, selectedSolution, onSelectSolution]);
  
  return (
    <svg ref={svgRef} className="w-full h-full"></svg>
  );
};

export default ParetoFront;