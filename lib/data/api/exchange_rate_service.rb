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
    Rails.env.development? ? call_dev() : call_prod()
  rescue => e
    sleep(TimeHelper.exponential_drop_off(base: ((RETRY_COUNT - retry_count) + 2)))
    retry_count -= 1
    retry if retry_count > 0
    ErrorHelper.mail_error('Exchange rate service error: ', e.message)
  end

  def call_prod
    prev_date = (Date.today - 1).to_s   # We are looking for the closing rate of the previous day.
    return if date_data_exists?(prev_date)

    uri = URI("https://openexchangerates.org/api/historical/#{prev_date}.json?app_id=#{Rails.application.secrets.exchange_key}&base=USD")
    service_request = Net::HTTP.get_response(uri)

    if service_request.response.is_a?(Net::HTTPSuccess)
      save_exchange_rate(JSON.parse(service_request.body).to_h['rates']['ZAR'], prev_date)
    else
      raise IntegrationError.new("Unable to successfully call openexchangerate.org, error: #{service_request.body.to_s}, code: #{service_request.response.code}")
    end
  end

  def call_dev
    prev_date = Date.today - 1
    start_date = ExchangeRate.order(date: :desc).first.date + 1

    return if start_date > prev_date

    (start_date..prev_date).each do |date|
      next if date_data_exists?(date)

      puts ('call exchange for: ' + date.to_s)

      uri = URI("https://openexchangerates.org/api/historical/#{date.to_s}.json?app_id=#{Rails.application.secrets.exchange_key}&base=USD")
      service_request = Net::HTTP.get_response(uri)

      if service_request.response.is_a?(Net::HTTPSuccess)
        save_exchange_rate(JSON.parse(service_request.body).to_h['rates']['ZAR'], date)
      else
        raise IntegrationError.new("Unable to successfully call openexchangerate.org, error: #{service_request.body.to_s}")
      end
    end
  end

  def save_exchange_rate(value, date)
    ExchangeRate.create(base: 'USD',
                        currency: 'ZAR',
                        rate: value,
                        date: date,
                        source: 'https://openexchangerates.org/')
  end

  def date_data_exists?(date)
    ExchangeRate.where(date: date, source: 'https://openexchangerates.org/').any?
  end
end