'use client'

import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts'

interface FraudTrendData {
  date: string
  fraudAttempts: number
  detected: number
  falsePositives: number
}

export function FraudTrends() {
  const { data: trendData, isLoading } = useQuery<FraudTrendData[]>({
    queryKey: ['fraud-trends'],
    queryFn: async () => {
      // Mock data - in production, fetch from API
      const last7Days = Array.from({ length: 7 }, (_, i) => {
        const date = new Date()
        date.setDate(date.getDate() - (6 - i))
        return {
          date: date.toISOString().split('T')[0],
          fraudAttempts: Math.floor(Math.random() * 50) + 20,
          detected: Math.floor(Math.random() * 40) + 15,
          falsePositives: Math.floor(Math.random() * 5) + 1
        }
      })
      return last7Days
    },
    refetchInterval: 60000, // Refresh every minute
  })

  if (isLoading) {
    return (
      <div className="bg-white shadow-md rounded-lg p-6 border border-gray-200 hover:shadow-lg transition-shadow duration-200">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-6 border border-gray-200 hover:shadow-lg transition-shadow duration-200">
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-900">
          Fraud Detection Trends
        </h3>
        <p className="text-sm text-gray-500">
          Last 7 days activity
        </p>
      </div>

      <div className="space-y-6">
        {/* Line chart for fraud attempts vs detected */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Detection Performance
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value, name) => [value, name === 'fraudAttempts' ? 'Fraud Attempts' : 'Detected']}
              />
              <Line 
                type="monotone" 
                dataKey="fraudAttempts" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="fraudAttempts"
              />
              <Line 
                type="monotone" 
                dataKey="detected" 
                stroke="#22c55e" 
                strokeWidth={2}
                name="detected"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Bar chart for false positives */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            False Positives
          </h4>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value) => [value, 'False Positives']}
              />
              <Bar 
                dataKey="falsePositives" 
                fill="#f97316" 
                radius={[2, 2, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-200">
          <div className="text-center">
            <div className="text-2xl font-semibold text-red-600">
              {trendData?.reduce((sum, day) => sum + day.fraudAttempts, 0) || 0}
            </div>
            <div className="text-xs text-gray-500">Total Attempts</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-semibold text-green-600">
              {trendData?.reduce((sum, day) => sum + day.detected, 0) || 0}
            </div>
            <div className="text-xs text-gray-500">Detected</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-semibold text-orange-600">
              {trendData?.reduce((sum, day) => sum + day.falsePositives, 0) || 0}
            </div>
            <div className="text-xs text-gray-500">False Positives</div>
          </div>
        </div>
      </div>
    </div>
  )
}
