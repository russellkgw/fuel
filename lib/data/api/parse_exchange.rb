# 201 success
# 400 if missing data
# 409 if record exists
# 500 in save fails
class Data::Api::ParseExchange

  def self.add_exchange_rate(data)
    api_data = new

    return 400 unless api_data.fields_present?(data)
    return 409 if api_data.record_exists?(data)
    api_data.add_exchange_rate(data) ? 201 : 500
  end

  def initialize
  end

  def fields_present?(data)
    data[:date].present? && data[:base].present? && data[:currency].present? && data[:value].present?
  end

  def record_exists?(data)
    ExchangeRate.where(date: data[:date], base: data[:base], currency: data[:currency]).first
  end

  def add_exchange_rate(data)
    ExchangeRate.new(data).save
  end
end