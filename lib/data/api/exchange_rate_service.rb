class Data::Api::ExchangeRateService
  class Error < RuntimeError; end
  class IntegrationError < Error; end

  include TimeHelper
  include ErrorHelper

  RETRY_COUNT = 3

  def self.call_service
    new.call_service
  end

  def call_service(retry_count: RETRY_COUNT)
    call()
  rescue => e
    sleep(TimeHelper.exponential_drop_off(base: ((RETRY_COUNT - retry_count) + 2)))
    retry_count -= 1
    retry if retry_count > 0
    ErrorHelper.mail_error('Exchange rate service error: ', e.message)
  end

  def call
    prev_date = (Date.today - 1).to_s   # We are looking for the closing rate of the previous day.
    return if ExchangeRate.where(date: prev_date).any?

    uri = URI("https://openexchangerates.org/api/historical/#{prev_date}.json?app_id=#{Rails.application.secrets.exchange_key}&base=USD")
    service_request = Net::HTTP.get_response(uri)

    if service_request.response.is_a?(Net::HTTPSuccess)
      save_exchange_rate(JSON.parse(service_request.body).to_h['rates']['ZAR'], prev_date)
    else
      raise IntegrationError.new("Unable to successfully call openexchangerate.org, error: #{service_request.body.to_s}")
    end
  end

  def save_exchange_rate(value, date)
    ExchangeRate.create(base: 'USD',
                        currency: 'ZAR',
                        rate: value,
                        date: date,
                        source: 'https://openexchangerates.org/')
  end
end