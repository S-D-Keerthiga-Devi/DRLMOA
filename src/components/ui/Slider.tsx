import React from 'react';

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  format?: (value: number) => string;
}

const Slider: React.FC<SliderProps> = ({
  label,
  value,
  onChange,
  min,
  max,
  step,
  suffix,
  format,
}) => {
  const displayValue = format ? format(value) : value.toString();
  
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <label className="block text-sm font-medium">{label}</label>
        <span className="text-sm font-medium">
          {displayValue} {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
      />
    </div>
  );
};

export default Slider;