export interface City {
  id: number;
  x: number;
  y: number;
}

export interface TSPData {
  name: string;
  type: string;
  dimension: number;
  cities: City[];
}

export class DataLoader {
  /**
   * Parses a TSP file from TSPLIB
   * @param contents The contents of the TSP file
   * @returns TSPData object with the parsed data
   */
  public static parseTSPFile(contents: string): TSPData {
    const lines = contents
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);
    
    const result: Partial<TSPData> = {
      cities: []
    };
    
    let readingNodes = false;
    let i = 0;
    
    while (i < lines.length) {
      const line = lines[i];
      
      if (line.startsWith('NAME')) {
        result.name = line.split(':')[1].trim();
      } else if (line.startsWith('TYPE')) {
        result.type = line.split(':')[1].trim();
      } else if (line.startsWith('DIMENSION')) {
        result.dimension = parseInt(line.split(':')[1].trim(), 10);
      } else if (line === 'NODE_COORD_SECTION') {
        readingNodes = true;
        i++;
        continue;
      } else if (line === 'EOF') {
        break;
      } else if (readingNodes) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 3) {
          const id = parseInt(parts[0], 10);
          const x = parseFloat(parts[1]);
          const y = parseFloat(parts[2]);
          
          result.cities!.push({ id, x, y });
        }
      }
      
      i++;
    }
    
    return result as TSPData;
  }
  
  /**
   * Combines two TSP datasets to create a bi-objective problem
   * @param dataA First TSP dataset (e.g., kroA100)
   * @param dataB Second TSP dataset (e.g., kroB100)
   * @returns Combined bi-objective TSP data
   */
  public static combineBiObjectiveData(dataA: TSPData, dataB: TSPData): any {
    if (dataA.dimension !== dataB.dimension) {
      throw new Error('Datasets must have the same dimension');
    }
    
    const combined: any = {
      name: `${dataA.name}-${dataB.name}`,
      dimension: dataA.dimension,
      cities: []
    };
    
    for (let i = 0; i < dataA.dimension; i++) {
      const cityA = dataA.cities[i];
      const cityB = dataB.cities[i];
      
      combined.cities.push({
        id: cityA.id,
        x1: cityA.x,
        y1: cityA.y,
        x2: cityB.x,
        y2: cityB.y
      });
    }
    
    return combined;
  }
  
  /**
   * Generates random bi-objective TSP instances
   * @param numCities Number of cities
   * @param numInstances Number of instances to generate
   * @param type Problem type (euclidean or mixed)
   * @returns Array of generated instances
   */
  public static generateInstances(
    numCities: number,
    numInstances: number = 1,
    type: 'euclidean' | 'mixed' = 'euclidean'
  ): any[] {
    const instances = [];
    
    for (let inst = 0; inst < numInstances; inst++) {
      const cities = [];
      
      for (let i = 0; i < numCities; i++) {
        if (type === 'euclidean') {
          // For Euclidean bi-objective TSP (4D input)
          cities.push({
            id: i,
            x1: Math.random(),
            y1: Math.random(),
            x2: Math.random(),
            y2: Math.random()
          });
        } else {
          // For Mixed bi-objective TSP (3D input)
          cities.push({
            id: i,
            x: Math.random(),
            y: Math.random(),
            z: Math.random()
          });
        }
      }
      
      instances.push({
        id: inst,
        numCities,
        type,
        cities
      });
    }
    
    return instances;
  }
  
  /**
   * Load standard TSPLIB test instances
   * This is a placeholder for actual file loading functionality
   */
  public static async loadTSPLIBInstance(name: string): Promise<TSPData> {
    // In a real implementation, this would load and parse the actual file
    // For now, we'll generate a dummy instance
    
    const dimension = parseInt(name.replace(/\D/g, ''), 10) || 100;
    
    const cities: City[] = [];
    for (let i = 0; i < dimension; i++) {
      cities.push({
        id: i + 1,
        x: Math.random() * 1000,
        y: Math.random() * 1000
      });
    }
    
    return {
      name,
      type: 'TSP',
      dimension,
      cities
    };
  }
}