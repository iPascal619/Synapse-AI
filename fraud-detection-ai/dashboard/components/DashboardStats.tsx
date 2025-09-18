'use client'

import { useQuery } from '@tanstack/react-query'
import { 
  CurrencyDollarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline'

interface DashboardMetrics {
  totalTransactions: number
  fraudDetected: number
  falsePositives: number
  averageResponseTime: number
  fraudRate: number
  accuracy: number
  systemStatus: {
    modelStatus: string
    fallbackMode: boolean
    uptime: number
    apiRequests: number
  }
  lastUpdated: string
}

export function DashboardStats() {
  const { data: metrics, isLoading, error } = useQuery<DashboardMetrics>({
    queryKey: ['dashboard-metrics'],
    queryFn: async () => {
      // Fetch real metrics from API
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/dashboard/metrics`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      return response.json()
    },
    refetchInterval: 10000, // Refresh every 10 seconds for real-time updates
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-white shadow-lg rounded-lg p-6 border border-gray-200 animate-pulse">
            <div className="h-20 bg-gray-200 rounded"></div>
          </div>
        ))}
      </div>
    )
  }

  if (error || !metrics) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">
          Failed to load metrics: {error?.message || 'Unknown error'}
        </p>
        <p className="text-sm text-red-500 mt-1">
          Please check if the API service is running.
        </p>
      </div>
    )
  }

  const stats = [
    {
      id: 1,
      name: 'Total Transactions',
      stat: metrics.totalTransactions.toLocaleString(),
      icon: CurrencyDollarIcon,
      change: metrics.systemStatus.apiRequests > 0 ? '+' + ((metrics.totalTransactions / Math.max(metrics.systemStatus.uptime, 1)) * 60).toFixed(1) + '/min' : 'N/A',
      changeType: 'neutral',
    },
    {
      id: 2,
      name: 'Fraud Detected',
      stat: metrics.fraudDetected.toString(),
      icon: ExclamationTriangleIcon,
      change: `${metrics.fraudRate}%`,
      changeType: metrics.fraudRate > 1.0 ? 'increase' : 'neutral',
    },
    {
      id: 3,
      name: 'Model Accuracy',
      stat: `${metrics.accuracy}%`,
      icon: CheckCircleIcon,
      change: metrics.systemStatus.fallbackMode ? 'Fallback Mode' : 'ML Models',
      changeType: metrics.systemStatus.fallbackMode ? 'decrease' : 'increase',
    },
    {
      id: 4,
      name: 'Response Time',
      stat: `${metrics.averageResponseTime}ms`,
      icon: ClockIcon,
      change: metrics.systemStatus.modelStatus === 'healthy' ? 'Optimal' : 
              metrics.systemStatus.modelStatus === 'degraded' ? 'Slow' : 'Issues',
      changeType: metrics.averageResponseTime < 50 ? 'increase' : 
                  metrics.averageResponseTime < 100 ? 'neutral' : 'decrease',
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {stats.map((item) => (
        <div
          key={item.id}
          className="bg-white shadow-lg rounded-lg p-6 border border-gray-200 hover:shadow-xl transition-shadow duration-300 relative overflow-hidden"
        >
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <item.icon
                className="h-8 w-8 text-gray-400"
                aria-hidden="true"
              />
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate uppercase tracking-wide">
                  {item.name}
                </dt>
                <dd className="flex items-baseline">
                  <div className="text-3xl font-bold text-gray-900 mb-2">
                    {item.stat}
                  </div>
                  <div
                    className={`ml-2 flex items-baseline text-xs font-semibold px-2 py-1 rounded-full ${
                      item.changeType === 'increase'
                        ? 'bg-green-100 text-green-800'
                        : item.changeType === 'decrease'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {item.change}
                  </div>
                </dd>
              </dl>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
