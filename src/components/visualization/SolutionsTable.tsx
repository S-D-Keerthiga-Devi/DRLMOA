import React from 'react';

interface SolutionsTableProps {
  instanceId: string;
  selectedSolution: number | null;
  onSelectSolution: (index: number | null) => void;
}

const SolutionsTable: React.FC<SolutionsTableProps> = ({ 
  instanceId,
  selectedSolution,
  onSelectSolution
}) => {
  // Generate some example data
  const baseCost1 = instanceId === 'kroAB100' ? 22000 : instanceId === 'kroAB150' ? 28000 : 34000;
  const baseCost2 = instanceId === 'kroAB100' ? 24000 : instanceId === 'kroAB150' ? 30000 : 36000;
  
  const solutions = Array.from({ length: 15 }, (_, i) => {
    const t = i / 14;  // Parameter between 0 and 1
    return {
      id: i,
      obj1: Math.round(baseCost1 * (1.0 + 0.2 * t) + Math.random() * 500),
      obj2: Math.round(baseCost2 * (1.2 - 0.2 * t) + Math.random() * 500),
    };
  });

  return (
    <div className="overflow-y-auto h-[340px]">
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">ID</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Obj 1</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Obj 2</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Sum</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
          {solutions.map(solution => (
            <tr 
              key={solution.id}
              className={`cursor-pointer hover:bg-blue-50 dark:hover:bg-blue-900/20 ${
                solution.id === selectedSolution ? 'bg-blue-100 dark:bg-blue-900/30' : ''
              }`}
              onClick={() => onSelectSolution(solution.id === selectedSolution ? null : solution.id)}
            >
              <td className="px-4 py-3 whitespace-nowrap text-sm font-medium">
                {solution.id + 1}
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-sm">
                {solution.obj1.toLocaleString()}
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-sm">
                {solution.obj2.toLocaleString()}
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-sm">
                {(solution.obj1 + solution.obj2).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default SolutionsTable;