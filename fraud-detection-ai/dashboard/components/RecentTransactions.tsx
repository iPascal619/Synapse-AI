'use client'

import { useQuery } from '@tanstack/react-query'
import { format } from 'date-fns'
import Link from 'next/link'

interface Transaction {
  id: string
  userId: string
  amount: number
  currency: string
  merchantId: string
  timestamp: string
  decision: 'APPROVE' | 'DENY' | 'REVIEW'
  riskScore: number
}

export function RecentTransactions() {
  const { data: transactions, isLoading } = useQuery<Transaction[]>({
    queryKey: ['recent-transactions'],
    queryFn: async () => {
      // Mock data - in production, fetch from API
      return [
        {
          id: 'tx_001',
          userId: 'user_123',
          amount: 2500.00,
          currency: 'USD',
          merchantId: 'merchant_abc',
          timestamp: new Date().toISOString(),
          decision: 'REVIEW',
          riskScore: 0.85
        },
        {
          id: 'tx_002', 
          userId: 'user_456',
          amount: 99.99,
          currency: 'USD',
          merchantId: 'merchant_def',
          timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
          decision: 'APPROVE',
          riskScore: 0.12
        },
        {
          id: 'tx_003',
          userId: 'user_789',
          amount: 15000.00,
          currency: 'USD', 
          merchantId: 'merchant_ghi',
          timestamp: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
          decision: 'DENY',
          riskScore: 0.94
        }
      ]
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  const getDecisionBadge = (decision: string, riskScore: number) => {
    switch (decision) {
      case 'DENY':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
            Denied
          </span>
        )
      case 'REVIEW':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-orange-100 text-orange-800">
            Review
          </span>
        )
      case 'APPROVE':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            Approved
          </span>
        )
      default:
        return null
    }
  }

  const getRiskScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-red-600'
    if (score >= 0.5) return 'text-orange-600'
    return 'text-green-600'
  }

  if (isLoading) {
    return (
      <div className="bg-white shadow-md rounded-lg p-6 border border-gray-200 hover:shadow-lg transition-shadow duration-200">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-4 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-6 border border-gray-200 hover:shadow-lg transition-shadow duration-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900">
          Recent Transactions
        </h3>
        <Link
          href="/review"
          className="text-sm text-blue-600 hover:text-blue-500"
        >
          View all
        </Link>
      </div>

      <div className="space-y-4">
        {transactions?.map((transaction) => (
          <div
            key={transaction.id}
            className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-900 truncate">
                  Transaction {transaction.id}
                </p>
                {getDecisionBadge(transaction.decision, transaction.riskScore)}
              </div>
              
              <div className="mt-1 flex items-center space-x-4 text-sm text-gray-500">
                <span>${transaction.amount.toLocaleString()}</span>
                <span>•</span>
                <span>{transaction.userId}</span>
                <span>•</span>
                <span>{format(new Date(transaction.timestamp), 'HH:mm:ss')}</span>
              </div>
            </div>

            <div className="ml-4 flex-shrink-0">
              <div className="text-right">
                <div className={`text-sm font-medium ${getRiskScoreColor(transaction.riskScore)}`}>
                  {(transaction.riskScore * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500">Risk</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {transactions && transactions.length === 0 && (
        <div className="text-center py-6">
          <p className="text-sm text-gray-500">No recent transactions</p>
        </div>
      )}
    </div>
  )
}
