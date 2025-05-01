import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface CityGraphProps {
  title?: string;
  numCities: number;
  showPath?: boolean;
  selectedCity?: number;
}

const CityGraph: React.FC<CityGraphProps> = ({ 
  title, 
  numCities,
  showPath = false,
  selectedCity,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current) return;
    
    // Clear previous graph
    d3.select(svgRef.current).selectAll("*").remove();
    
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const padding = 40;
    
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);
    
    // Generate random cities
    const cities = Array.from({ length: numCities }, () => ({
      x: Math.random() * (width - 2 * padding) + padding,
      y: Math.random() * (height - 2 * padding) + padding,
    }));
    
    // Generate a realistic TSP path
    let path: number[] = [];
    
    if (showPath) {
      // Simple nearest neighbor path for demonstration
      const visited = new Set<number>();
      let current = 0; // Start with the first city
      path.push(current);
      visited.add(current);
      
      while (visited.size < numCities) {
        let bestDist = Infinity;
        let bestCity = -1;
        
        for (let i = 0; i < numCities; i++) {
          if (!visited.has(i)) {
            const dist = Math.sqrt(
              Math.pow(cities[current].x - cities[i].x, 2) +
              Math.pow(cities[current].y - cities[i].y, 2)
            );
            
            if (dist < bestDist) {
              bestDist = dist;
              bestCity = i;
            }
          }
        }
        
        if (bestCity !== -1) {
          current = bestCity;
          path.push(current);
          visited.add(current);
        } else {
          break;
        }
      }
      
      // Complete the cycle
      path.push(path[0]);
    }
    
    // Draw edges if showing path
    if (showPath) {
      const lineGenerator = d3.line<{x: number, y: number}>()
        .x(d => d.x)
        .y(d => d.y);
      
      svg.append("path")
        .datum(path.map(i => cities[i]))
        .attr("d", lineGenerator)
        .attr("fill", "none")
        .attr("class", "edge solution");
    }
    
    // Draw cities
    svg.selectAll("circle")
      .data(cities)
      .enter()
      .append("circle")
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", 6)
      .attr("class", (_, i) => `node ${i === selectedCity ? 'selected' : ''}`)
      .append("title")
      .text((_, i) => `City ${i+1}`);
    
    // Add city labels for a subset of cities
    if (numCities <= 100) {
      const labelStep = Math.max(1, Math.floor(numCities / 20));
      
      svg.selectAll("text")
        .data(cities.filter((_, i) => i % labelStep === 0))
        .enter()
        .append("text")
        .attr("x", d => d.x + 8)
        .attr("y", d => d.y + 4)
        .text((_, i) => `${(i * labelStep) + 1}`)
        .attr("font-size", "10px")
        .attr("fill", "currentColor");
    }
    
    // Add title if provided
    if (title) {
      svg.append("text")
        .attr("x", width / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "14px")
        .attr("font-weight", "500")
        .text(title);
    }
  }, [numCities, showPath, selectedCity, title]);
  
  return (
    <div className="w-full h-full flex items-center justify-center">
      <svg ref={svgRef} className="w-full h-full"></svg>
    </div>
  );
};

export default CityGraph;